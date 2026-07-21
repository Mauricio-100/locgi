# train_words.py - Entraînement de locgi par Mots (V3)
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import re
from safetensors.torch import save_file

# --- ARCHITECTURE ADVANCED ---
class GopuBrainAdvanced(nn.Module):
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

def create_batches(data, batch_size, seq_len):
    n_batches = len(data) // (batch_size * seq_len)
    data = data[:n_batches * batch_size * seq_len]
    data = torch.tensor(data, dtype=torch.long).view(batch_size, -1)
    batches = []
    for i in range(0, data.size(1) - seq_len, seq_len):
        x = data[:, i : i+seq_len]
        y = data[:, i+1 : i+seq_len+1]
        batches.append((x, y))
    return batches

print("🚀 GOPU.INC - Entraînement Word-Level")

# 1. Charger le vocabulaire
with open("vocab.json", "r", encoding="utf-8") as f:
    word_to_int = json.load(f)
vocab_size = len(word_to_int)

# 2. Charger et Tokeniser le corpus (exactement comme le générateur)
with open("corpus.txt", "r", encoding="utf-8") as f:
    texte = f.read()

print("⏳ Découpage du texte en mots...")
mots = re.findall(r'\w+|[^\w\s]', texte.lower())

# Convertir les mots en index mathématiques. 
# Si un mot est inconnu (ce qui n'arrive pas ici mais par sécurité), on met l'index de <UNK>
data_indices = [word_to_int.get(mot, word_to_int["<UNK>"]) for mot in mots]

# 3. Préparer l'entraînement
batch_size = 16
seq_len = 12 # Le modèle va regarder 12 mots en arrière pour prédire le suivant
batches = create_batches(data_indices, batch_size, seq_len)

# On augmente un peu la taille des embeddings car un mot est plus complexe qu'une lettre
model = GopuBrainAdvanced(vocab_size, embed_dim=128, hidden_dim=256)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.002)
criterion = nn.CrossEntropyLoss()

print(f"📊 Taille du vocabulaire : {vocab_size} tokens")
print(f"📚 Total des mots à apprendre : {len(data_indices)}")
print("=" * 50)

# 4. Boucle d'entraînement
epochs = 30
for epoch in range(epochs):
    model.train()
    total_loss = 0
    hidden = None
    
    for x, y in batches:
        optimizer.zero_grad()
        if hidden is not None:
            hidden = tuple([h.detach() for h in hidden])
        
        out, hidden = model(x, hidden)
        loss = criterion(out.reshape(-1, vocab_size), y.reshape(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    if len(batches) > 0:
        avg_loss = total_loss / len(batches)
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Époque {epoch:3d}/{epochs} | Loss: {avg_loss:.4f}")

# 5. Sauvegarde
save_file(model.state_dict(), "gopu_poids.safetensors")
print("=" * 50)
print("✅ Entraînement terminé ! Nouveaux poids générés.")

