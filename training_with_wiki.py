# enrichir_corpus.py - Ajoute plus de phrases au corpus existant
import random

# Nouvelles phrases à ajouter
nouvelles_phrases = [
    # Plus de salutations
    "hey comment ca va.",
    "coucou tout le monde.",
    "bonjour a tous.",
    "salut les amis.",
    "bonne journee.",
    "passe une bonne soiree.",
    
    # Plus de questions
    "quest ce que tu fais.",
    "tu aimes quoi.",
    "quel est ton hobby.",
    "tu parles francais.",
    "tu comprends le langage naturel.",
    "comment fonctionne un reseau de neurones.",
    "cest quoi le machine learning.",
    
    # Plus de réponses
    "je suis un modele de langage.",
    "je peux discuter avec toi.",
    "japprends de nouvelles choses.",
    "je suis entraine sur des donnees.",
    "je comprends le texte.",
    
    # Plus de faits
    "le soleil est une etoile de type naine jaune.",
    "la terre tourne autour du soleil.",
    "la lune est le satellite naturel de la terre.",
    "lunivers est immense.",
    "la galaxie andromede est proche de la notre.",
    
    # Plus de tech
    "le cloud computing est pratique.",
    "les donnees sont stockees dans le cloud.",
    "la cybersecurite est importante.",
    "les algorithmes sont partout.",
    "lopen source permet la collaboration.",
    
    # Dialogues utiles
    "tu es utile pour quoi.",
    "je suis la pour aider et discuter.",
    "tu as des limites.",
    "oui je suis un modele simple.",
    "tu apprends comment.",
    "japprends a partir de texte.",
]

# Lire le corpus existant
with open("corpus.txt", "r", encoding="utf-8") as f:
    corpus_existant = [ligne.strip() for ligne in f if ligne.strip()]

# Ajouter les nouvelles phrases
corpus_complet = corpus_existant + nouvelles_phrases

# Mélanger
random.shuffle(corpus_complet)

# Sauvegarder
with open("corpus.txt", "w", encoding="utf-8") as f:
    for phrase in corpus_complet:
        f.write(phrase + "\n")

print(f"Corpus enrichi !")
print(f"Ancien: {len(corpus_existant)} phrases")
print(f"Nouveau: {len(corpus_complet)} phrases")
print(f"Ajouté: {len(nouvelles_phrases)} nouvelles phrases")
