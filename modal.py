import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import re
import wikipedia
from safetensors.torch import load_file

# Configuration langue Wikipedia
wikipedia.set_lang("fr")

# 1. Architecture du cerveau GopuBrainAdvanced (inchangée)
class GopuBrainAdvanced(nn.Module):
    def __init__(self, vocab_size=1227, embed_dim=128, hidden_dim=256, num_layers=2):
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

# 2. Chargement du vocabulaire WORD-LEVEL
# ⚠️ Ce vocab.json doit être celui généré par train_vocab.py / utilisé par train_words.py
# (souvent nommé vocab_words.json à la base -> renomme-le en vocab.json, ou adapte le chemin ici)
with open("vocab.json", "r", encoding="utf-8") as f:
    word_to_int = json.load(f)
int_to_word = {i: w for w, i in word_to_int.items()}
vocab_size = len(word_to_int)
UNK_ID = word_to_int.get("<UNK>", 0)

model = GopuBrainAdvanced(vocab_size=vocab_size, embed_dim=128, hidden_dim=256)

try:
    model.load_state_dict(load_file("gopu_poids.safetensors"))
    model.eval()
    print("✅ Moteur Gopu.inc chargé avec succès.")
except Exception as e:
    print(f"❌ Erreur de chargement : {e}")

# Tokenizer identique à celui utilisé pour l'entraînement (train_words.py / train_vocab.py)
def tokeniser(texte):
    return re.findall(r"\w+|[^\w\s]", texte.lower())

# Ponctuation qui ne doit pas avoir d'espace avant elle
PONCTUATION_COLLEE = {".", ",", "!", "?", ";", ":", "'"}

def detokeniser(tokens):
    """Recolle une liste de mots/ponctuation en une phrase lisible."""
    phrase = ""
    for tok in tokens:
        if tok in PONCTUATION_COLLEE or not phrase:
            phrase += tok
        else:
            phrase += " " + tok
    return phrase

# 3. Fonction de génération, en word-level désormais
def generer_reponse(prompt, max_len=40, temperature=0.7):
    prompt_formate = f"question : {prompt} réponse :"
    tokens_prompt = tokeniser(prompt_formate)
    input_indices = [word_to_int.get(t, UNK_ID) for t in tokens_prompt]

    tokens_generes = []

    with torch.no_grad():
        hidden = None
        input_tensor = torch.tensor([input_indices], dtype=torch.long)
        for _ in range(max_len):
            logits, hidden = model(input_tensor, hidden)
            logits = logits[0, -1] / temperature
            probs = torch.softmax(logits, dim=-1)
            prochain_idx = torch.multinomial(probs, 1).item()

            mot = int_to_word.get(prochain_idx, "<UNK>")

            if mot in ("<PAD>",):
                break

            tokens_generes.append(mot)

            # Condition d'arrêt : fin de phrase après quelques mots
            if mot in (".", "!", "?") and len(tokens_generes) > 2:
                break

            # On ne repasse que le dernier token dans le LSTM stateful (plus rapide,
            # cohérent avec l'entraînement stateful)
            input_tensor = torch.tensor([[prochain_idx]], dtype=torch.long)

    return detokeniser(tokens_generes).strip()

# Boucle principale
print("\n" + "=" * 50)
print("🤖 MOTEUR GOPU.INC ACTIF (STABLE)")
print("=" * 50)

while True:
    user_input = input("\n[Toi] > ").strip()
    if not user_input:
        continue
    if user_input.lower() in ["quit", "exit"]:
        break

    if user_input.startswith("wiki:"):
        sujet = user_input.split("wiki:", 1)[1].strip()
        try:
            print(f"🔍 {wikipedia.summary(sujet, sentences=1)}")
        except Exception:
            print("[Wiki] > Sujet non trouvé.")
    else:
        print(f"[Gopu] > {generer_reponse(user_input)}")

