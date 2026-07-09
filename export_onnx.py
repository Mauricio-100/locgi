import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from safetensors.torch import load_file

# 1. Définition de l'architecture (Doit être identique à ton script d'entraînement)
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

print("🚀 GOPU.INC - Préparation de l'exportation ONNX")

# 2. Chargement dynamique du vocabulaire
with open("vocab.json", "r", encoding="utf-8") as f:
    vocab = json.load(f)
vocab_size = len(vocab)
print(f"✅ Vocabulaire détecté : {vocab_size} tokens")

# 3. Initialisation du modèle
model = GopuBrainAdvanced(vocab_size, embed_dim=128, hidden_dim=256)

# 4. Chargement des poids
try:
    state_dict = load_file("gopu_poids.safetensors")
    model.load_state_dict(state_dict)
    model.eval()
    print("✅ Poids chargés avec succès dans l'architecture.")
except Exception as e:
    print(f"❌ Erreur lors du chargement des poids : {e}")
    exit()

# 5. Exportation en ONNX
# On crée un "dummy input" pour simuler une séquence de 12 mots (batch size 1)
dummy_input = torch.randint(0, vocab_size, (1, 12), dtype=torch.long)

try:
    torch.onnx.export(
        model,
        dummy_input,
        "gopu_brain.onnx",
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size', 1: 'sequence_length'},
                      'output': {0: 'batch_size', 1: 'sequence_length'}}
    )
    print("✨ Succès : Le fichier 'gopu_brain.onnx' est prêt pour Node.js !")
except Exception as e:
    print(f"❌ Erreur lors de l'exportation : {e}")

