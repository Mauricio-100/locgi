import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os
from safetensors.torch import save_file

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

def create_stateful_batches(texte, char_to_int, batch_size, seq_len):
    data = [char_to_int.get(c, 0) for c in texte]
    n_batches = len(data) // (batch_size * seq_len)
    data = data[:n_batches * batch_size * seq_len]
    data = torch.tensor(data, dtype=torch.long).view(batch_size, -1)
    batches = []
    for i in range(0, data.size(1) - seq_len, seq_len):
        x = data[:, i : i+seq_len]
        y = data[:, i+1 : i+seq_len+1]
        batches.append((x, y))
    return batches

# Lecture du corpus existant
with open("corpus.txt", "r", encoding="utf-8") as f:
    texte = f.read()

vocab = sorted(list(set(texte)))
char_to_int = {c: i for i, c in enumerate(vocab)}
with open("vocab.json", "w", encoding="utf-8") as f:
    json.dump(char_to_int, f)

vocab_size = len(vocab)
model = GopuBrainAdvanced(vocab_size)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.002)
criterion = nn.CrossEntropyLoss()
batches = create_stateful_batches(texte, char_to_int, 16, 32)

print(f"🚀 Entraînement sur {len(texte)} caractères...")

for epoch in range(100):
    model.train()
    total_loss = 0
    hidden = None
    for x, y in batches:
        optimizer.zero_grad()
        if hidden is not None: hidden = tuple(h.detach() for h in hidden)
        logits, hidden = model(x, hidden)
        loss = criterion(logits.reshape(-1, vocab_size), y.reshape(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if epoch % 10 == 0:
        print(f"Époque {epoch} | Loss: {total_loss/len(batches):.4f}")

save_file(model.state_dict(), "gopu_poids.safetensors")
print("✅ Entraînement terminé et sauvegardé.")

