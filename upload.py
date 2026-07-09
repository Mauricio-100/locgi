# upload.py - Version corrigée
import os
from huggingface_hub import HfApi, create_repo, upload_file

# Récupérer ton nom d'utilisateur automatiquement
from huggingface_hub import whoami
try:
    user_info = whoami()
    USERNAME = user_info['name']
    print(f"✅ Connecté en tant que: {USERNAME}")
except:
    USERNAME = input("Entre ton nom d'utilisateur Hugging Face: ")

REPO_ID = f"{USERNAME}/locgi"

print(f"📦 Repository: {REPO_ID}")

# Créer le repo
api = HfApi()
try:
    create_repo(
        repo_id=REPO_ID,
        repo_type="model",
        exist_ok=True,
        private=False
    )
    print(f"✅ Repo {REPO_ID} créé/vérifié")
except Exception as e:
    print(f"⚠️ Le repo existe peut-être déjà: {e}")

# Upload des fichiers
files = [
    "chat.py",
    "train_simple.py", 
    "train_batch.py",
    "make_corpus.py",
    "enrichir_corpus.py",
    "gopu_poids.safetensors",
    "vocab.json",
    "corpus.txt",
]

print("\n📤 Upload en cours...")

for file in files:
    if os.path.exists(file):
        try:
            upload_file(
                path_or_fileobj=file,
                path_in_repo=file,
                repo_id=REPO_ID,
                repo_type="model"
            )
            print(f"  ✅ {file}")
        except Exception as e:
            print(f"  ❌ {file} - {e}")
    else:
        print(f"  ⚠️ {file} - Fichier non trouvé")

print(f"\n🎉 Upload terminé !")
print(f"👉 https://huggingface.co/{REPO_ID}")
