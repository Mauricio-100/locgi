import torch
import torch.nn as nn
import numpy as np
import json
import os
from safetensors.torch import save_file

class GopuBrain(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        x = self.dropout(x)
        out, hidden = self.lstm(x, hidden)
        return self.fc(out), hidden

# ==========================================
# 1. CHARGEMENT DES DONNÉES
# ==========================================

def load_corpus():
    with open("corpus.txt", "r", encoding="utf-8") as f:
        return f.read()

print("Chargement du corpus...")
texte = load_corpus()
print(f"Taille du corpus : {len(texte)} caractères")

# Construction du vocabulaire
vocab = sorted(list(set(texte)))
char_to_int = {c: i for i, c in enumerate(vocab)}
int_to_char = {i: c for i, c in enumerate(vocab)}

with open("vocab.json", "w") as f:
    json.dump(char_to_int, f)

print(f"Taille du vocabulaire : {len(vocab)} caractères")

# ==========================================
# 2. PARAMÈTRES D'ENTRAÎNEMENT
# ==========================================

vocab_size = len(vocab)
embed_dim = 128
hidden_dim = 256
num_layers = 2
sequence_length = 100  # Longueur de séquence

model = GopuBrain(vocab_size, embed_dim, hidden_dim, num_layers)
optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
criterion = nn.CrossEntropyLoss()

# ==========================================
# 3. FONCTION D'ENTRAÎNEMENT CORRIGÉE
# ==========================================

def train_epoch():
    """Entraîne sur une époque complète"""
    # Convertir tout le texte en indices
    data = torch.tensor([char_to_int[c] for c in texte], dtype=torch.long)
    
    total_loss = 0
    n_batches = 0
    
    # Parcourir le texte par séquences
    for i in range(0, len(data) - sequence_length, sequence_length // 2):
        # Prendre une séquence et la suivante
        x = data[i:i+sequence_length].unsqueeze(0)
        y = data[i+1:i+sequence_length+1].unsqueeze(0)
        
        model.train()
        optimizer.zero_grad()
        
        out, _ = model(x)
        # Utiliser reshape au lieu de view
        loss = criterion(out.reshape(-1, vocab_size), y.reshape(-1))
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / n_batches if n_batches > 0 else float('inf')

# ==========================================
# 4. ENTRAÎNEMENT
# ==========================================

print("\nDébut de l'entraînement...")
print("=" * 50)

epochs = 300
best_loss = float('inf')

for epoch in range(epochs):
    loss = train_epoch()
    
    if loss < best_loss:
        best_loss = loss
        save_file(model.state_dict(), "gopu_poids.safetensors")
    
    if epoch % 30 == 0 or epoch == epochs - 1:
        print(f"Époque {epoch:3d}/{epochs} | Loss: {loss:.4f} | Best: {best_loss:.4f}")

print("=" * 50)
print("✓ Entraînement terminé !")
print(f"✓ Meilleure loss : {best_loss:.4f}")
print(f"✓ Modèle sauvegardé dans 'gopu_poids.safetensors'")

# ==========================================
# 5. TEST RAPIDE
# ==========================================

def chat_quick(prompt, max_chars=10):
    model.eval()
    result = prompt
    
    for _ in range(max_chars):
        data = [char_to_int.get(c, 0) for c in result[-sequence_length:]]
        x = torch.tensor(data, dtype=torch.long).unsqueeze(0)
        out, _ = model(x)
        prochain_char = int_to_char[torch.argmax(out[0, -1]).item()]
        result += prochain_char
    
    return result

print("\nTests rapides:")
print(f"  bonj -> {chat_quick('bonj')}")
print(f"  le f -> {chat_quick('le f')}")
print(f"  comm -> {chat_quick('comm')}")
print(f"  python -> {chat_quick('python')}")
