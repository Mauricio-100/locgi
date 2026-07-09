import torch
from safetensors.torch import load_file
from create_model_and_train.py import GopuBrainAdvanced # Remplace par le nom de ton fichier

# 1. Charger les poids réels
poids = load_file("gopu_poids.safetensors")

# 2. Créer une instance du modèle avec les dimensions que tu soupçonnes
# D'après ton erreur, tes poids ont : embed_dim=64, hidden_dim=128
model = GopuBrainAdvanced(vocab_size=94, embed_dim=64, hidden_dim=128)
modele_params = model.state_dict()

print("--- COMPARAISON DES DIMENSIONS ---")
for nom, tenseur in poids.items():
    if nom in modele_params:
        if tenseur.shape != modele_params[nom].shape:
            print(f"❌ Mismatch sur {nom}: Poids={tenseur.shape} | Code={modele_params[nom].shape}")
        else:
            print(f"✅ {nom} correspond : {tenseur.shape}")
    else:
        print(f"⚠️ {nom} trouvé dans le fichier mais pas dans le code.")

