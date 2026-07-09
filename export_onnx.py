# export_onnx.py - Convertisseur PyTorch vers ONNX pour gopu.inc
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from safetensors.torch import load_file

print("🚀 GOPU.INC - Préparation de l'exportation ONNX")

# 1. On redéfinit l'architecture exacte pour charger les poids
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
        return self.fc2(out), hidden

# 2. Charger le vocabulaire
try:
    with open("vocab.json", "r", encoding="utf-8") as f:
        vocab = json.load(f)
    vocab_size = len(vocab)
except FileNotFoundError:
    print("❌ Erreur : vocab.json introuvable.")
    exit()

# 3. Charger le modèle entraîné
model = GopuBrainAdvanced(vocab_size, embed_dim=128, hidden_dim=256)
model.load_state_dict(load_file("gopu_poids.safetensors"))
model.eval() # TRÈS IMPORTANT : Désactive les couches de Dropout pour l'exportation

# 4. Créer des entrées factices (Dummy Inputs)
# ONNX a besoin de voir "comment" la donnée traverse le réseau pour le compiler.
# On lui donne un faux mot (index 0) et une mémoire LSTM vide.
dummy_input = torch.tensor([[0]], dtype=torch.long)
dummy_h0 = torch.zeros(2, 1, 256) # num_layers=2, batch=1, hidden_dim=256
dummy_c0 = torch.zeros(2, 1, 256)
dummy_hidden = (dummy_h0, dummy_c0)

# 5. Exportation magique vers ONNX
print("⏳ Compilation du modèle en ONNX... (Cela peut prendre quelques secondes)")
torch.onnx.export(
    model, 
    (dummy_input, dummy_hidden), # Les entrées que le modèle attend
    "gopu_brain.onnx",           # Le nom du fichier final
    export_params=True,          # Exporter avec les poids entraînés
    opset_version=14,            # Version standard de compatibilité ONNX
    do_constant_folding=True,    # Optimisation mathématique
    input_names=['input_token', 'hidden_in_h', 'hidden_in_c'],
    output_names=['output_logits', 'hidden_out_h', 'hidden_out_c'],
    dynamic_axes={               # Permet de donner des phrases de n'importe quelle longueur plus tard
        'input_token': {0: 'batch_size', 1: 'sequence_length'},
        'output_logits': {0: 'batch_size', 1: 'sequence_length'}
    }
)

print("=" * 50)
print("✅ SUCCÈS : 'gopu_brain.onnx' a été généré !")
print("Ce fichier contient toute ton IA. Il est prêt à être lu par Node.js.")
print("=" * 50)

