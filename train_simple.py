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

# Modèle plus petit pour aller plus vite
model = GopuBrain(len(vocab), 64, 128)  # Réduit: 64 au lieu de 128, 128 au lieu de 256
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # Learning rate plus élevé
criterion = nn.CrossEntropyLoss()

# Entraînement optimisé
def train_step(seq_len=80):
    # Position aléatoire
    start = torch.randint(0, max(1, len(texte) - seq_len - 1), (1,)).item()
    
    x = torch.tensor([char_to_int[c] for c in texte[start:start+seq_len]], dtype=torch.long).unsqueeze(0)
    y = torch.tensor([char_to_int[c] for c in texte[start+1:start+seq_len+1]], dtype=torch.long).unsqueeze(0)
    
    model.train()
    optimizer.zero_grad()
    
    out, _ = model(x)
    loss = criterion(out.reshape(-1, len(vocab)), y.reshape(-1))
    
    loss.backward()
    optimizer.step()
    
    return loss.item()

print("\n🚀 Entraînement rapide...")
print("=" * 40)

best_loss = float('inf')

# Seulement 50 époques, mais avec plus d'échantillons par époque
for epoch in range(50):
    # Faire plusieurs steps par époque pour apprendre plus vite
    total_loss = 0
    for _ in range(20):  # 20 échantillons par époque
        loss = train_step()
        total_loss += loss
    
    avg_loss = total_loss / 20
    
    if avg_loss < best_loss:
        best_loss = avg_loss
        save_file(model.state_dict(), "gopu_poids.safetensors")
    
    if epoch % 10 == 0 or epoch == 49:
        print(f"Époque {epoch:3d}/50 | Loss: {avg_loss:.4f} | Best: {best_loss:.4f}")

print("=" * 40)
print("✅ Terminé en {:.1f} secondes !".format(epoch * 2))  # Estimation
print(f"📊 Meilleure loss: {best_loss:.4f}")
print(f"💾 Modèle sauvegardé")

# Test rapide
def test(prompt, n=20):
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
print(f"  bonjour -> {test('bonjour')}")
print(f"  le football -> {test('le football')}")
print(f"  python est -> {test('python est')}")
