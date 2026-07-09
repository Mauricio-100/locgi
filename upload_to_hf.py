from huggingface_hub import HfApi, create_repo

# Configuration
REPO_ID = "Mauricio-100/locgi"
FILES_TO_UPLOAD = [
    "gopu_poids.safetensors",
    "vocab.json",
    "config.json"
]

api = HfApi()

try:
    # 1. Créer le repo si besoin
    create_repo(REPO_ID, private=True, exist_ok=True)
    print(f"✅ Dépôt {REPO_ID} prêt.")

    # 2. Upload des fichiers
    for file in FILES_TO_UPLOAD:
        print(f"📤 Envoi de {file}...")
        api.upload_file(
            path_or_fileobj=file,
            path_in_repo=file,
            repo_id=REPO_ID,
        )
    
    print("🚀 Félicitations ! Ton modèle est sur le Hub : https://huggingface.co/" + REPO_ID)

except Exception as e:
    print(f"❌ Erreur : {e}")

