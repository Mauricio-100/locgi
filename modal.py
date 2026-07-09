# modal.py - Version corrigée pour GopuBrainAdvanced
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import wikipedia
from safetensors.torch import load_file

wikipedia.set_lang("fr")

# 1. Utilisation obligatoire de l'architecture avancée
class GopuBrainAdvanced(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128, num_layers=2):
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

# Initialisation des vocabulaires
with open("vocab.json", "r", encoding="utf-8") as f:
    char_to_int = json.load(f)
int_to_char = {i: c for c, i in char_to_int.items()}
vocab_size = len(char_to_int)

# Chargement du modèle avancé
model = GopuBrainAdvanced(vocab_size)
model.load_state_dict(load_file("gopu_poids.safetensors"))
model.eval()

def generer_reponse(prompt, max_len=100, temperature=0.6):
    # Appliquer le template exact du corpus d'entraînement
    prompt_formate = f"Question : {prompt} Réponse : "
    
    input_indices = [char_to_int.get(c, 0) for c in prompt_formate]
    caractere_precedent = ""
    reponse_generee = ""
    
    with torch.no_grad():
        for _ in range(max_len):
            # On passe toute la séquence générée jusqu'ici pour garder le contexte du Conv1d
            input_tensor = torch.tensor([input_indices], dtype=torch.long)
            logits, _ = model(input_tensor)
            
            # On prend la prédiction du tout dernier caractère
            logits = logits[0, -1] / temperature
            probs = torch.softmax(logits, dim=-1)
            
            # Légère pénalité de répétition immédiate
            if caractere_precedent:
                idx_precedent = char_to_int.get(caractere_precedent, 0)
                if idx_precedent < len(probs):
                    probs[idx_precedent] *= 0.3
            
            prochain_char_idx = torch.multinomial(probs, 1).item()
            char = int_to_char[prochain_char_idx]
            
            # Condition d'arrêt : fin de ligne ou point final après une phrase minimale
            if char == "\n" or (char in ".!?" and len(reponse_generee) > 10):
                if char in ".!?":
                    reponse_generee += char
                break
                
            reponse_generee += char
            caractere_precedent = char
            input_indices.append(prochain_char_idx)
            
    return reponse_generee.strip()

print("\n" + "=" * 50)
print("🤖 MOTEUR GOPU.INC ACTIF (ADVANCED MODE V2)")
print("=" * 50)
print("Tape 'wiki: sujet' pour Wikipedia")
print("Tape 'quit' ou 'exit' pour quitter")
print("=" * 50)

while True:
    user_input = input("\n[Toi] > ").strip()
    
    if not user_input:
        continue
    
    if user_input.lower() in ["quit", "exit"]:
        print("[Gopu] > Au revoir ! 👋")
        break
    
    if user_input.startswith("wiki:"):
        sujet = user_input.split("wiki:", 1)[1].strip()
        if not sujet:
            print("[Wiki] > Spécifie un sujet.")
            continue
        print(f"🔍 Recherche Wikipedia sur : {sujet}")
        try:
            summary = wikipedia.summary(sujet, sentences=2)
            print(f"[Wiki] > {summary}")
        except:
            print("[Wiki] > Sujet non trouvé.")
    else:
        reponse = generer_reponse(user_input)
        print(f"[Gopu] > {reponse}")

