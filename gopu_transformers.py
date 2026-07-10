"""
gopu_transformers.py
Gestionnaire d'inférence/entraînement "façon transformers" pour la famille
de modèles Gopu/locgi.

Usage :

    from gopu_transformers import GopuBrainAdvanced, ModelGopu, Models

    model = GopuBrainAdvanced(
        model_name=ModelGopu("locgi"),   # télécharge Mauricio-100/locgi si absent
        TypeModel=Models("gopu"),
    )
    # ou plus court, la résolution se fait automatiquement :
    model = GopuBrainAdvanced("locgi")          # cherche en local, sinon télécharge
    model = GopuBrainAdvanced("./locgi")        # dossier local explicite

    print(model.inference("Quelle est la capitale de la France ?"))
    model.train(corpus_path="corpus_large.txt", model_retrain=True, epochs=60)

    model.tenseurs()        # liste les tenseurs et leurs formes
    model.neurone()         # nb approximatif d'unités/neurones
    model.memory()          # empreinte mémoire des poids
    model.CalculeParam()    # nb de paramètres entraînables
"""

import os
import re
import json
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from safetensors.torch import load_file, save_file
except ImportError as e:
    raise ImportError("pip install safetensors --break-system-packages") from e


# ============================================================
# Architecture brute — identique à celle utilisée dans
# train_words.py / modal.py, pour rester compatible avec les
# poids .safetensors déjà entraînés.
# ============================================================
class _GopuBrainAdvancedCore(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.conv1d = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.norm = nn.LayerNorm(hidden_dim)
        self.attention_weights = nn.Linear(hidden_dim, 1)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, hidden=None):
        embedded = self.embedding(x)
        conv_in = embedded.permute(0, 2, 1)
        conv_out = F.relu(self.conv1d(conv_in)).permute(0, 2, 1)
        lstm_in = embedded + conv_out
        out, hidden = self.lstm(lstm_in, hidden)
        out = self.norm(out)
        attn_dist = torch.softmax(self.attention_weights(out), dim=1)
        out = out + (out * attn_dist)
        out = F.gelu(self.fc1(out))
        return self.fc2(self.dropout(out)), hidden


# Registre d'architectures : un seul point d'entrée pour toutes les
# familles de modèles. Ajoute une entrée ici quand tu crées "pref",
# "capus", "cmo", etc.
MODEL_REGISTRY = {
    "gopu": _GopuBrainAdvancedCore,
    # "pref": PrefBrainCore,
    # "capus": CapusBrainCore,
    # "cmo": CMOBrainCore,
}


def Models(type_nom: str) -> str:
    """Sélectionne une famille d'architecture, ex: Models("gopu")."""
    type_nom = type_nom.lower()
    if type_nom not in MODEL_REGISTRY:
        raise ValueError(
            f"Type de modèle inconnu : '{type_nom}'. Disponibles : {list(MODEL_REGISTRY.keys())}"
        )
    return type_nom


# ============================================================
# Résolution du modèle : local d'abord, sinon téléchargement HF
# ============================================================
LOCGI_HOME = Path(os.environ.get("LOCGI_HOME", Path.home() / ".locgi"))
LOCGI_MODELS_DIR = LOCGI_HOME / "lib" / "model"
HF_ORG = "Mauricio-100"
FICHIERS_REQUIS = ["config.json", "vocab.json", "gopu_poids.safetensors"]


def ModelGopu(model_name: str) -> Path:
    """
    Résout le chemin local d'un modèle :
      1. si model_name est déjà un dossier existant (ex: "./locgi") -> utilisé tel quel
      2. sinon, cherche dans ~/.locgi/lib/model/{model_name}
      3. si absent ou incomplet, télécharge Mauricio-100/{model_name} depuis
         huggingface.co et le place dans ~/.locgi/lib/model/{model_name}
    """
    chemin_direct = Path(model_name)
    if chemin_direct.exists() and chemin_direct.is_dir():
        return chemin_direct

    chemin_local = LOCGI_MODELS_DIR / model_name
    if chemin_local.exists() and all((chemin_local / f).exists() for f in FICHIERS_REQUIS):
        return chemin_local

    print(f"⬇️  Modèle '{model_name}' absent en local, téléchargement depuis "
          f"huggingface.co/{HF_ORG}/{model_name} ...")
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise ImportError(
            "huggingface_hub requis pour le téléchargement automatique : "
            "pip install huggingface_hub --break-system-packages"
        ) from e

    chemin_local.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=f"{HF_ORG}/{model_name}",
        local_dir=str(chemin_local),
        local_dir_use_symlinks=False,
    )
    print(f"✅ Modèle téléchargé dans {chemin_local}")
    return chemin_local


def GetConfig(chemin_modele) -> dict:
    chemin_config = Path(chemin_modele) / "config.json"
    if not chemin_config.exists():
        raise FileNotFoundError(f"config.json introuvable dans {chemin_modele}")
    with open(chemin_config, "r", encoding="utf-8") as f:
        return json.load(f)


def _tokeniser(texte: str):
    return re.findall(r"\w+|[^\w\s]", texte.lower())


_PONCTUATION_COLLEE = {".", ",", "!", "?", ";", ":", "'"}


def _detokeniser(tokens):
    phrase = ""
    for tok in tokens:
        if tok in _PONCTUATION_COLLEE or not phrase:
            phrase += tok
        else:
            phrase += " " + tok
    return phrase


# ============================================================
# Classe principale — API façon transformers (AutoModel-like)
# ============================================================
class GopuBrainAdvanced:
    def __init__(self, model_name, hidden_dim=None, vocab_size=None, embed_dim=None,
                 TypeModel=None, device="cpu"):
        self.type_modele = TypeModel or Models("gopu")
        self.device = device

        self.chemin = model_name if isinstance(model_name, Path) else ModelGopu(str(model_name))

        # Les arguments explicites surchargent le config.json du modèle
        config_disque = GetConfig(self.chemin)
        self.config = {
            "vocab_size": vocab_size or config_disque["vocab_size"],
            "embed_dim": embed_dim or config_disque["embed_dim"],
            "hidden_dim": hidden_dim or config_disque["hidden_dim"],
            "num_layers": config_disque.get("num_layers", 2),
        }

        with open(self.chemin / "vocab.json", "r", encoding="utf-8") as f:
            self.word_to_int = json.load(f)
        self.int_to_word = {i: w for w, i in self.word_to_int.items()}
        self.PAD_ID = self.word_to_int.get("<PAD>", 0)
        self.UNK_ID = self.word_to_int.get("<UNK>", 0)

        Architecture = MODEL_REGISTRY[self.type_modele]
        self.model = Architecture(
            vocab_size=self.config["vocab_size"],
            embed_dim=self.config["embed_dim"],
            hidden_dim=self.config["hidden_dim"],
            num_layers=self.config["num_layers"],
        ).to(self.device)

        poids_path = self.chemin / "gopu_poids.safetensors"
        if poids_path.exists():
            self.model.load_state_dict(load_file(str(poids_path)))
            print(f"✅ Poids chargés depuis {poids_path}")
        else:
            print(f"⚠️ Aucun poids trouvé dans {self.chemin}, modèle initialisé aléatoirement.")

    # --------------------------------------------------------
    # Introspection
    # --------------------------------------------------------
    def tenseurs(self) -> dict:
        """Liste tous les tenseurs du modèle avec leur forme."""
        infos = {}
        for nom, tenseur in self.model.state_dict().items():
            infos[nom] = tuple(tenseur.shape)
            print(f"{nom:30s} {tuple(tenseur.shape)}")
        return infos

    def neurone(self) -> int:
        """Nombre approximatif d'unités (neurones) réparties sur les couches."""
        total = (
            self.config["embed_dim"]
            + self.config["hidden_dim"] * self.config["num_layers"]
            + self.config["hidden_dim"]
            + self.config["vocab_size"]
        )
        print(f"🧠 ~{total:,} neurones (unités) répartis sur les couches")
        return total

    def memory(self) -> float:
        """Empreinte mémoire des poids en Mo (float32)."""
        n_params = sum(p.numel() for p in self.model.parameters())
        mo = n_params * 4 / (1024 ** 2)
        print(f"💾 {n_params:,} paramètres -> ~{mo:.2f} Mo en float32")
        return mo

    def CalculeParam(self) -> int:
        """Nombre de paramètres entraînables."""
        n_entrainables = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.model.parameters())
        print(f"📐 Paramètres entraînables : {n_entrainables:,} / {n_total:,} au total")
        return n_entrainables

    # --------------------------------------------------------
    # Inférence
    # --------------------------------------------------------
    def inference(self, prompt: str, max_len=40, temperature=0.7) -> str:
        prompt_formate = f"question : {prompt} réponse :"
        indices = [self.word_to_int.get(t, self.UNK_ID) for t in _tokeniser(prompt_formate)]

        tokens_generes = []
        self.model.eval()
        with torch.no_grad():
            hidden = None
            x = torch.tensor([indices], dtype=torch.long, device=self.device)
            for _ in range(max_len):
                logits, hidden = self.model(x, hidden)
                logits = logits[0, -1] / temperature
                probs = torch.softmax(logits, dim=-1)
                idx_suivant = torch.multinomial(probs, 1).item()
                mot = self.int_to_word.get(idx_suivant, "<UNK>")
                if mot == "<PAD>":
                    break
                tokens_generes.append(mot)
                if mot in (".", "!", "?") and len(tokens_generes) > 2:
                    break
                x = torch.tensor([[idx_suivant]], dtype=torch.long, device=self.device)

        return _detokeniser(tokens_generes).strip()

    # --------------------------------------------------------
    # Entraînement / ré-entraînement
    # --------------------------------------------------------
    def train(self, corpus_path="corpus.txt", model_retrain=True, epochs=60,
              batch_size=32, lr=0.001, save_best=True):
        """
        Entraîne (ou ré-entraîne) le modèle exemple par exemple : chaque ligne
        "Question : ... Réponse : ..." est une séquence indépendante, le hidden
        state du LSTM repart de zéro à chaque batch (pas de mélange entre
        questions sans rapport, contrairement à l'ancien train_words.py V3).
        """
        if not model_retrain:
            print("ℹ️ model_retrain=False, entraînement ignoré.")
            return self

        with open(corpus_path, "r", encoding="utf-8") as f:
            lignes = [l.strip() for l in f if l.strip()]

        exemples = []
        for ligne in lignes:
            if "Réponse :" not in ligne:
                continue
            indices = [self.word_to_int.get(t, self.UNK_ID) for t in _tokeniser(ligne)]
            if len(indices) >= 2:
                exemples.append(indices)

        print(f"📚 {len(exemples)} exemples chargés depuis {corpus_path}")

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(ignore_index=self.PAD_ID)
        best_loss = float("inf")

        for epoch in range(epochs):
            self.model.train()
            batches = self._faire_batches(exemples, batch_size)
            total_loss = 0.0
            for x, y in batches:
                optimizer.zero_grad()
                out, _ = self.model(x, None)
                loss = criterion(out.reshape(-1, self.config["vocab_size"]), y.reshape(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / max(len(batches), 1)
            if save_best and avg_loss < best_loss:
                best_loss = avg_loss
                save_file(self.model.state_dict(), str(self.chemin / "gopu_poids.safetensors"))

            if epoch % 5 == 0 or epoch == epochs - 1:
                print(f"Époque {epoch:3d}/{epochs} | Loss: {avg_loss:.4f}")

        print(f"✅ Entraînement terminé. Meilleure loss : {best_loss:.4f}")
        return self

    def _faire_batches(self, exemples, batch_size):
        exemples_tries = sorted(exemples, key=len)
        lots = [exemples_tries[i:i + batch_size] for i in range(0, len(exemples_tries), batch_size)]
        random.shuffle(lots)
        batches = []
        for lot in lots:
            max_len = max(len(e) for e in lot)
            if max_len < 2:
                continue
            x_batch = torch.full((len(lot), max_len - 1), self.PAD_ID, dtype=torch.long)
            y_batch = torch.full((len(lot), max_len - 1), self.PAD_ID, dtype=torch.long)
            for j, e in enumerate(lot):
                L = len(e) - 1
                x_batch[j, :L] = torch.tensor(e[:-1], dtype=torch.long)
                y_batch[j, :L] = torch.tensor(e[1:], dtype=torch.long)
            batches.append((x_batch, y_batch))
        return batches


if __name__ == "__main__":
    # Exemple d'usage rapide
    model = GopuBrainAdvanced("./locgi", TypeModel=Models("gopu"))
    model.CalculeParam()
    model.memory()
    model.neurone()
    print(model.inference("bonjour"))

