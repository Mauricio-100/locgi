from huggingface_hub import HfApi

# Configuration
REPO_ID = "Mauricio-100/locgi"
api = HfApi()

def clean_repo():
    print(f"🧹 Connexion au dépôt {REPO_ID} pour nettoyage...")
    
    try:
        # 1. Lister tous les fichiers
        files = api.list_repo_files(repo_id=REPO_ID)
        
        # 2. Supprimer les fichiers (sauf .gitattributes qui est crucial)
        for file in files:
            if file == ".gitattributes":
                print(f"⏩ Ignoré : {file}")
                continue
            
            print(f"🗑️ Suppression de {file}...")
            api.delete_file(path_in_repo=file, repo_id=REPO_ID)
            
        print("✅ Nettoyage terminé avec succès !")
        
    except Exception as e:
        print(f"❌ Erreur lors du nettoyage : {e}")

if __name__ == "__main__":
    confirm = input(f"Es-tu sûr de vouloir supprimer tous les fichiers de {REPO_ID} ? (oui/non) : ")
    if confirm.lower() == "oui":
        clean_repo()
    else:
        print("Opération annulée.")

