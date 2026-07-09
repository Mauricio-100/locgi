import os
import subprocess
from github import Github

# Configuration
REPO_NAME = "Mauricio-100/locgi" # Note: Ton nom d'utilisateur est Mauricio-100
TOKEN = os.getenv("GITHUB_TOKEN")

if not TOKEN:
    print("❌ Erreur : GITHUB_TOKEN non défini. Fais 'export GITHUB_TOKEN=ton_token'")
    exit()

g = Github(TOKEN)

def setup_github():
    print(f"🚀 Connexion à GitHub pour {REPO_NAME}...")
    
    # 1. Créer le dépôt sur GitHub si nécessaire
    try:
        user = g.get_user()
        try:
            repo = user.get_repo("locgi")
            print("✅ Dépôt trouvé sur GitHub.")
        except:
            print("⏳ Création du dépôt sur GitHub...")
            repo = user.create_repo("locgi", private=True)
            print("✅ Dépôt créé.")
    except Exception as e:
        print(f"❌ Erreur GitHub : {e}")
        return

    # 2. Commandes Git locales
    print("📦 Préparation des fichiers locaux...")
    try:
        # Initialiser git si besoin
        if not os.path.exists(".git"):
            subprocess.run(["git", "init"], check=True)
            subprocess.run(["git", "branch", "-M", "main"], check=True)
        
        # Ajouter le remote
        remote_url = f"https://{TOKEN}@github.com/{REPO_NAME}.git"
        subprocess.run(["git", "remote", "remove", "origin"], stderr=subprocess.DEVNULL) # Nettoie si déjà là
        subprocess.run(["git", "remote", "add", "origin", remote_url], check=True)
        
        # Committer et pousser
        subprocess.run(["git", "add", "."], check=True)
        subprocess.run(["git", "commit", "-m", "Auto-push depuis locgi script"], check=True)
        subprocess.run(["git", "push", "-u", "origin", "main"], check=True)
        
        print("🚀 Succès : Tout ton projet est sur GitHub !")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Erreur Git : {e}")

if __name__ == "__main__":
    setup_github()

