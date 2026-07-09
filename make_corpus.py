# make_corpus.py - Version avec Wikipedia améliorée
import json
import wikipedia
import random
import re

wikipedia.set_lang("fr")

# ==========================================
# 1. CORPUS MANUEL ÉTOFFÉ
# ==========================================

# Niveau 1 : Bases (phrases très simples)
niveau_1 = [
    "locgi apprend.",
    "gopu.inc innove.",
    "le code tourne.",
    "pytorch calcule.",
    "un tenseur change.",
    "python est simple.",
    "locgi est une ia.",
    "le systeme avance.",
    "bonjour tout le monde.",
    "comment ca va aujourdhui.",
    "le soleil brille.",
    "la lune est belle.",
    "les etoiles scintillent.",
    "le vent souffle.",
    "la pluie tombe.",
    "le chat dort.",
    "le chien court.",
    "les oiseaux chantent.",
    "le jardin est vert.",
    "la mer est bleue.",
]

# Niveau 2 : Intermédiaire (concepts et actions)
niveau_2 = [
    "locgi est un modele de langage rapide.",
    "gopu.inc developpe des outils de securite.",
    "le reseau de neurones ajuste ses poids.",
    "les fichiers safetensors stockent les tenseurs.",
    "un developpeur ecrit du code en python.",
    "l'apprentissage par cursus facilite la convergence du modele.",
    "le fichier de configuration valide les donnees.",
    "locgi est developpe par Mauricio Mangituka en 2026.",
    "un modele de langage comprend le texte.",
    "le deep learning utilise des reseaux de neurones.",
    "les donnees d'entrainement sont importantes.",
    "la performance du modele s'ameliorer avec le temps.",
    "le football est un sport collectif populaire.",
    "le basket se joue avec un ballon et un panier.",
    "la musique adoucit les moeurs.",
    "l'art est une forme d'expression humaine.",
    "la science explore l'univers.",
    "la technologie evolue rapidement.",
    "les ordinateurs sont partout.",
    "internet connecte le monde.",
]

# Niveau 3 : Complexe (phrases longues avec contexte)
niveau_3 = [
    "le modele locgi analyse chaque caractere pour predire la suite du texte.",
    "gopu.inc cree des architectures intelligentes adaptees aux environnements de developpement.",
    "les matrices imbriquees calculent les probabilites de chaque lettre du dictionnaire unique.",
    "optimiser les hyperparametres permet d'eviter les minima locaux pendant l'entrainement progressif.",
    "une intelligence artificielle peut apprendre a partir de donnees textuelles structurees.",
    "le traitement du langage naturel permet aux machines de comprendre le langage humain.",
    "les reseaux de neurones recurrents sont adaptes pour les sequences de texte.",
    "le machine learning transforme la facon dont nous interagissons avec la technologie.",
    "les donnees de qualite sont essentielles pour un apprentissage efficace.",
    "la puissance de calcul des GPU accelere l'entrainement des modeles.",
]

# Niveau 4 : Dialogues et questions/réponses
niveau_4 = [
    "bonjour comment ca va.",
    "ca va bien merci et toi.",
    "je vais bien merci de demander.",
    "quelle est la capitale de la france.",
    "paris est la capitale de la france.",
    "combien font deux plus deux.",
    "deux plus deux font quatre.",
    "qui est le president de la france.",
    "emmanuel macron est le president de la france.",
    "j'aime la programmation.",
    "python est un langage de programmation puissant.",
    "pytorch est une bibliotheque pour l'apprentissage profond.",
    "comment s'appelle le soleil.",
    "le soleil s'appelle le soleil.",
    "pourquoi le ciel est bleu.",
    "le ciel est bleu a cause de la diffusion de la lumiere.",
    "ou se trouve la tour eiffel.",
    "la tour eiffel se trouve a paris.",
    "quand est ne locgi.",
    "locgi est ne en 2026.",
    "que signifie gopu.inc.",
    "gopu.inc est une entreprise qui developpe des ia.",
]

# ==========================================
# 2. RÉCUPÉRATION DE DONNÉES WIKIPEDIA AMÉLIORÉE
# ==========================================

def clean_text(text):
    """Nettoie le texte Wikipedia"""
    # Enlever les références [1], [2], etc.
    text = re.sub(r'\[\d+\]', '', text)
    # Enlever les parenthèses inutiles
    text = re.sub(r'\([^)]*\)', '', text)
    # Remplacer les sauts de ligne
    text = text.replace('\n', ' ')
    return text

def get_wikipedia_data(sujets, sentences=4):
    """Récupère des résumés Wikipedia avec meilleure gestion des erreurs"""
    data = []
    
    # Mapping des sujets pour mieux chercher
    mapping = {
        "intelligence artificielle": ["intelligence artificielle", "IA"],
        "apprentissage profond": ["apprentissage profond", "deep learning"],
        "reseau de neurones": ["réseau de neurones", "neurone"],
        "python": ["Python", "langage python"],
        "technologie": ["technologie", "innovation"],
        "science": ["science", "sciences"],
        "informatique": ["informatique", "ordinateur"],
        "football": ["football", "sport"],
        "paris": ["Paris", "ville de Paris"],
        "programmation": ["programmation", "code informatique"],
    }
    
    for sujet in sujets:
        trouve = False
        
        # Essayer avec le sujet original puis ses variantes
        sujets_a_tester = [sujet]
        if sujet in mapping:
            sujets_a_tester.extend(mapping[sujet])
        
        for test_sujet in sujets_a_tester:
            try:
                print(f"  Essai: {test_sujet}...")
                page = wikipedia.page(test_sujet)
                summary = page.summary
                
                # Nettoyer
                summary = clean_text(summary)
                
                # Découper en phrases
                phrases = summary.split('. ')
                phrases = [p.strip() for p in phrases if len(p.strip()) > 20]
                
                # Prendre les N premières phrases
                for phrase in phrases[:sentences]:
                    if len(phrase) > 10:
                        # S'assurer que la phrase finit par un point
                        if not phrase.endswith('.'):
                            phrase += '.'
                        data.append(phrase.lower())
                
                print(f"  ✓ Wikipedia: {sujet} ajouté ({len(phrases[:sentences])} phrases)")
                trouve = True
                break
                
            except wikipedia.exceptions.DisambiguationError as e:
                # Si ambigu, prendre le premier résultat
                try:
                    premier = e.options[0]
                    print(f"  → Ambigu, essai: {premier}")
                    page = wikipedia.page(premier)
                    summary = page.summary
                    summary = clean_text(summary)
                    phrases = summary.split('. ')
                    phrases = [p.strip() for p in phrases if len(p.strip()) > 20]
                    for phrase in phrases[:sentences]:
                        if len(phrase) > 10:
                            if not phrase.endswith('.'):
                                phrase += '.'
                            data.append(phrase.lower())
                    print(f"  ✓ Wikipedia: {sujet} ajouté via {premier}")
                    trouve = True
                    break
                except:
                    pass
                    
            except wikipedia.exceptions.PageError:
                print(f"  ✗ Page non trouvée: {test_sujet}")
                
            except Exception as e:
                print(f"  ✗ Erreur: {e}")
        
        if not trouve:
            print(f"  ✗ Wikipedia: {sujet} non trouvé")
    
    return data

# Liste des sujets Wikipedia à récupérer
sujets_wiki = [
    "intelligence artificielle",
    "apprentissage profond",
    "reseau de neurones",
    "python",
    "technologie",
    "science",
    "informatique",
    "football",
    "paris",
    "programmation",
]

# ==========================================
# 3. ASSEMBLAGE DU CORPUS FINAL
# ==========================================

print("Récupération des données Wikipedia...")
print("-" * 40)
donnees_wiki = get_wikipedia_data(sujets_wiki, sentences=3)
print("-" * 40)

# Assemblage complet
corpus_complet = niveau_1 + niveau_2 + niveau_3 + niveau_4 + donnees_wiki

# Mélanger un peu
random.shuffle(corpus_complet)

# Écriture dans le fichier
nom_fichier = "corpus.txt"
with open(nom_fichier, "w", encoding="utf-8") as f:
    for phrase in corpus_complet:
        f.write(phrase + "\n")

print("\n" + "=" * 50)
print(f"[SUCCÈS] Fichier '{nom_fichier}' généré !")
print(f"Nombre total d'exemples : {len(corpus_complet)}")
print(f"  - Niveau 1 (Bases)          : {len(niveau_1)}")
print(f"  - Niveau 2 (Intermédiaire)  : {len(niveau_2)}")
print(f"  - Niveau 3 (Complexe)       : {len(niveau_3)}")
print(f"  - Niveau 4 (Dialogues)      : {len(niveau_4)}")
print(f"  - Wikipedia                 : {len(donnees_wiki)}")
print("=" * 50)

# Afficher quelques exemples Wikipedia
if donnees_wiki:
    print("\nExemples de données Wikipedia ajoutées:")
    for i in range(min(3, len(donnees_wiki))):
        print(f"  {i+1}. {donnees_wiki[i][:80]}...")
