# train_simple.py - Version ultra-simple
import torch
import torch.nn as nn
import json
from safetensors.torch import save_file

class GopuBrain(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        out, hidden = self.lstm(x, hidden)
        return self.fc(out), hidden

# Charger le corpus
with open("corpus.txt", "r", encoding="utf-8") as f:
    texte = f.read()

print(f"📚 Corpus: {len(texte)} caractères")

# Vocabulaire
vocab = sorted(list(set(texte)))
char_to_int = {c: i for i, c in enumerate(vocab)}
int_to_char = {i: c for i, c in enumerate(vocab)}

with open("vocab.json", "w") as f:
    json.dump(char_to_int, f)

print(f"📝 Vocabulaire: {len(vocab)} caractères")

# Modèle
model = GopuBrain(len(vocab), 128, 256)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
criterion = nn.CrossEntropyLoss()

# Entraînement
def train_step(seq_len=10):
    # Choisir une position aléatoire
    start = torch.randint(0, max(1, len(texte) - seq_len - 1), (1,)).item()
    
    # Préparer les données
    x = torch.tensor([char_to_int[c] for c in texte[start:start+seq_len]], dtype=torch.long).unsqueeze(0)
    y = torch.tensor([char_to_int[c] for c in texte[start+1:start+seq_len+1]], dtype=torch.long).unsqueeze(0)
    
    model.train()
    optimizer.zero_grad()
    
    out, _ = model(x)
    loss = criterion(out.reshape(-1, len(vocab)), y.reshape(-1))
    
    loss.backward()
    optimizer.step()
    
    return loss.item()

print("\n🚀 Entraînement en cours...")
print("=" * 50)

best_loss = float('inf')

for epoch in range(5000):
    loss = train_step()
    
    if loss < best_loss:
        best_loss = loss
        save_file(model.state_dict(), "gopu_poids.safetensors")
    
    if epoch % 50 == 0 or epoch == 4999:
        print(f"Époque {epoch:4d}/5000 | Loss: {loss:.4f} | Best: {best_loss:.4f}")

print("=" * 50)
print("✅ Entraînement terminé !")
print(f"📊 Meilleure loss: {best_loss:.4f}")
print(f"💾 Modèle sauvegardé dans 'gopu_poids.safetensors'")

# Test
def test(prompt, n=15):
    model.eval()
    result = prompt
    for _ in range(n):
        data = [char_to_int.get(c, 0) for c in result[-50:]]
        x = torch.tensor(data, dtype=torch.long).unsqueeze(0)
        out, _ = model(x)
        char = int_to_char[torch.argmax(out[0, -1]).item()]
        result += char
    return result

print("\n🧪 Tests:")
print(f"  bonj -> {test('bonj')}")
print(f"  le f -> {test('le f')}")
print(f"  comm -> {test('comm')}")
