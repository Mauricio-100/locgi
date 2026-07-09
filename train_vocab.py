# train_vocab.py - Générateur de Vocabulaire (Approche par Mots)
import json
import re
from collections import Counter

print("🚀 GOPU.INC - Création du Vocabulaire (Word-Level)")
print("=" * 50)

# 1. Charger le corpus complet
try:
    with open("corpus.txt", "r", encoding="utf-8") as f:
        texte = f.read()
except FileNotFoundError:
    print("❌ Erreur : Le fichier 'corpus.txt' est introuvable.")
    exit()

# 2. Tokenization intelligente (Regex)
# \w+ attrape les mots entiers (lettres/chiffres)
# [^\w\s] attrape la ponctuation (!, ?, ., etc.) en ignorant les espaces
print("⏳ Découpage du texte en mots et ponctuation...")
tokens_bruts = re.findall(r'\w+|[^\w\s]', texte.lower())

# 3. Compter les occurrences
compteur = Counter(tokens_bruts)
mots_uniques = list(compteur.keys())

print(f"📊 Total des mots dans le corpus : {len(tokens_bruts)}")
print(f"📊 Total des mots UNIQUES trouvés : {len(mots_uniques)}")

# 4. Ajout des "Tokens Spéciaux" (Crucial pour les vraies IA)
# <PAD> : Pour remplir les vides si une phrase est trop courte
# <UNK> : (Unknown) Si un utilisateur tape un mot que le modèle ne connaît pas
vocab = ["<PAD>", "<UNK>"] 
vocab.extend(mots_uniques)

# 5. Création du dictionnaire { "mot": index_mathematique }
word_to_int = {word: i for i, word in enumerate(vocab)}

# 6. Sauvegarde du nouveau vocabulaire
with open("vocab_words.json", "w", encoding="utf-8") as f:
    json.dump(word_to_int, f, ensure_ascii=False, indent=4)

print("=" * 50)
print(f"✅ Succès ! Nouveau vocabulaire sauvegardé dans 'vocab_words.json'")
print(f"Taille finale du vocabulaire : {len(vocab)} tokens.")
print("=" * 50)

