"""
colab_train_locgi_v2.py

Version adaptée pour GitHub Actions (pas de !shell, token via env)
"""

# %% [1. Installation]
import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])

install("torch")
install("safetensors")
install("huggingface_hub")
install("pandas")
install("pyarrow")
install("requests")

# %% [2. Imports]
import os
import re
import json
import random
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from safetensors.torch import save_file
from huggingface_hub import hf_hub_download, upload_file, login

REPO_ID = "Mauricio-100/locgi"

# %% [3. Authentification HuggingFace]
HF_TOKEN = os.environ.get("HF_TOKEN")
if HF_TOKEN:
    login(HF_TOKEN)
    print("✅ Authentifié via HF_TOKEN")
else:
    print("❌ HF_TOKEN non trouvé dans les variables d'environnement")
    exit(1)

# %% [4. Télécharger le corpus existant depuis ton repo HF]
print("📥 Téléchargement de corpus.txt depuis le Hub...")
corpus_path = hf_hub_download(repo_id=REPO_ID, filename="corpus.txt")
with open(corpus_path, "r", encoding="utf-8") as f:
    corpus_existant = f.read()
print(f"✅ Corpus existant : {len(corpus_existant)} caractères")

# %% [5. Télécharger frenchSTS et extraire des phrases]
print("📥 Téléchargement de frenchSTS...")
STS_URL = "https://huggingface.co/datasets/CATIE-AQ/frenchSTS/resolve/main/data/test-00000-of-00001.parquet"
response = requests.get(STS_URL)
with open("frenchSTS_test.parquet", "wb") as f:
    f.write(response.content)
print("✅ Téléchargé")

df = pd.read_parquet("frenchSTS_test.parquet")
print("Colonnes :", list(df.columns))

def clean_sentence(s):
    s = str(s).strip()
    s = re.sub(r"\s+", " ", s)
    return s

col_candidates = [c for c in df.columns if "sent" in c.lower() or "text" in c.lower()]
phrases_sts = set()
for col in col_candidates:
    for val in df[col].dropna():
        phrase = clean_sentence(val)
        if 15 < len(phrase) < 200:
            if not phrase.endswith((".", "!", "?")):
                phrase += "."
            phrases_sts.add(phrase)

print(f"✅ {len(phrases_sts)} phrases extraites de frenchSTS")

# %% [6. Fusionner et sauvegarder le corpus enrichi]
lignes_existantes = [l.strip() for l in corpus_existant.split("\n") if l.strip()]
corpus_complet = lignes_existantes + list(phrases_sts)
random.shuffle(corpus_complet)

with open("corpus.txt", "w", encoding="utf-8") as f:
    for ligne in corpus_complet:
        f.write(ligne + "\n")

print(f"📊 Corpus final : {len(corpus_complet)} lignes "
      f"({len(lignes_existantes)} existantes + {len(phrases_sts)} nouvelles)")

# %% [7. Reconstruire le vocabulaire (word-level)]
with open("corpus.txt", "r", encoding="utf-8") as f:
    texte_total = f.read()

tokens_bruts = re.findall(r"\w+|[^\w\s]", texte_total.lower())
mots_uniques = sorted(set(tokens_bruts))

vocab = ["<PAD>", "<UNK>"] + mots_uniques
word_to_int = {w: i for i, w in enumerate(vocab)}
int_to_word = {i: w for w, i in word_to_int.items()}
vocab_size = len(vocab)

with open("vocab.json", "w", encoding="utf-8") as f:
    json.dump(word_to_int, f, ensure_ascii=False)

print(f"📝 Vocabulaire reconstruit : {vocab_size} tokens")

# %% [8. Architecture : GopuTransformerLite (attention + MoE)]
class LightSelfAttention(nn.Module):
    def __init__(self, hidden_dim, n_heads=8):
        super().__init__()
        assert hidden_dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=True)

    def forward(self, x):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(mask, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(out)


class LightMoE(nn.Module):
    def __init__(self, hidden_dim, n_experts=8, ffn_mult=2):
        super().__init__()
        self.router = nn.Linear(hidden_dim, n_experts, bias=True)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * ffn_mult),
                nn.GELU(),
                nn.Linear(hidden_dim * ffn_mult, hidden_dim),
            ) for _ in range(n_experts)
        ])

    def forward(self, x):
        weights = F.softmax(self.router(x), dim=-1)
        top1 = torch.argmax(weights, dim=-1)
        out = torch.zeros_like(x)
        for e_id, expert in enumerate(self.experts):
            mask = (top1 == e_id).unsqueeze(-1)
            if mask.any():
                out = out + mask * expert(x)
        return out


class GopuBlockMoE(nn.Module):
    def __init__(self, hidden_dim, n_heads=8, n_experts=8):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = LightSelfAttention(hidden_dim, n_heads)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.moe = LightMoE(hidden_dim, n_experts)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.moe(self.norm2(x))
        return x


class GopuTransformerLite(nn.Module):
    def __init__(self, vocab_size, hidden_dim=256, n_layers=4, n_heads=8, n_experts=8, max_seq_len=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embedding = nn.Embedding(max_seq_len, hidden_dim)
        self.blocks = nn.ModuleList([
            GopuBlockMoE(hidden_dim, n_heads, n_experts) for _ in range(n_layers)
        ])
        self.norm_out = nn.LayerNorm(hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        B, T = x.shape
        positions = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.embedding(x) + self.pos_embedding(positions)
        for block in self.blocks:
            h = block(h)
        h = self.norm_out(h)
        return self.fc_out(h)

# %% [9. Préparer les données d'entraînement]
data_indices = [word_to_int.get(t, word_to_int["<UNK>"]) for t in tokens_bruts]

def create_batches(data, batch_size, seq_len):
    n_batches = len(data) // (batch_size * seq_len)
    data = data[:n_batches * batch_size * seq_len]
    data = torch.tensor(data, dtype=torch.long).view(batch_size, -1)
    batches = []
    for i in range(0, data.size(1) - seq_len, seq_len):
        x = data[:, i:i + seq_len]
        y = data[:, i + 1:i + seq_len + 1]
        batches.append((x, y))
    return batches

SEQ_LEN = 32
BATCH_SIZE = 32
batches = create_batches(data_indices, BATCH_SIZE, SEQ_LEN)
print(f"📚 {len(batches)} batches de taille {BATCH_SIZE}x{SEQ_LEN}")

# %% [10. Entraînement]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"💻 Device : {device}")

model = GopuTransformerLite(
    vocab_size=vocab_size, hidden_dim=256, n_layers=4,
    n_heads=8, n_experts=8, max_seq_len=SEQ_LEN
).to(device)

n_params = sum(p.numel() for p in model.parameters())
print(f"📊 {n_params:,} paramètres (~{n_params*4/1024/1024:.1f} Mo en float32)")

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
criterion = nn.CrossEntropyLoss(ignore_index=word_to_int["<PAD>"])

EPOCHS = 1000
best_loss = float("inf")

for epoch in range(EPOCHS):
    model.train()
    random.shuffle(batches)
    total_loss = 0
    for x, y in batches:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out.reshape(-1, vocab_size), y.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(batches)
    if avg_loss < best_loss:
        best_loss = avg_loss
        save_file(model.state_dict(), "gopu_poids.safetensors")

    print(f"Époque {epoch+1:3d}/{EPOCHS} | Loss: {avg_loss:.4f} | Best: {best_loss:.4f}")

print(f"✅ Entraînement terminé. Meilleure loss : {best_loss:.4f}")

# %% [11. Sauvegarder config.json]
config = {
    "model_type": "gopu-brain",
    "architecture": "GopuTransformerLite",
    "vocab_size": vocab_size,
    "hidden_dim": 256,
    "n_layers": 4,
    "n_heads": 8,
    "n_experts": 8,
    "max_seq_len": SEQ_LEN,
}
with open("config.json", "w", encoding="utf-8") as f:
    json.dump(config, f, indent=2)

# %% [12. Upload sur HuggingFace]
FILES_TO_UPLOAD = ["gopu_poids.safetensors", "vocab.json", "config.json", "corpus.txt"]

for file in FILES_TO_UPLOAD:
    print(f"📤 Envoi de {file}...")
    upload_file(
        path_or_fileobj=file,
        path_in_repo=file,
        repo_id=REPO_ID,
    )

print(f"🎉 Terminé ! Modèle disponible sur https://huggingface.co/{REPO_ID}")

# %% [13. Test rapide de génération]
def generer(prompt, max_len=30, temperature=0.8):
    model.eval()
    tokens = re.findall(r"\w+|[^\w\s]", f"question : {prompt} réponse :".lower())
    indices = [word_to_int.get(t, word_to_int["<UNK>"]) for t in tokens]
    generes = []
    with torch.no_grad():
        for _ in range(max_len):
            x = torch.tensor([indices[-SEQ_LEN:]], dtype=torch.long, device=device)
            out = model(x)
            logits = out[0, -1] / temperature
            probs = torch.softmax(logits, dim=-1)
            idx = torch.multinomial(probs, 1).item()
            mot = int_to_word.get(idx, "<UNK>")
            if mot == "<PAD>":
                break
            generes.append(mot)
            indices.append(idx)
            if mot in (".", "!", "?") and len(generes) > 2:
                break
    phrase = ""
    for tok in generes:
        phrase += tok if (tok in {".", ",", "!", "?"} or not phrase) else " " + tok
    return phrase

print("\n🧪 Tests de génération :")
for q in ["Quelle est la capitale de la France ?", "Qui a peint la Joconde ?"]:
    print(f"[Toi] > {q}")
    print(f"[Locgi] > {generer(q)}\n")
