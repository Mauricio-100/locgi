# chat.py - Version améliorée avec arrêt intelligent
import torch
import torch.nn as nn
import json
import wikipedia
from safetensors.torch import load_file

wikipedia.set_lang("fr")

class GopuBrain(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        out, hidden = self.lstm(x, hidden)
        return self.fc(out), hidden

# Initialisation
with open("vocab.json", "r") as f:
    char_to_int = json.load(f)
int_to_char = {i: c for c, i in char_to_int.items()}
vocab_size = len(char_to_int)

model = GopuBrain(vocab_size, 64, 128)
model.load_state_dict(load_file("gopu_poids.safetensors"))
model.eval()

def generer_reponse(prompt, max_len=80, temperature=0.7):
    """Génère une réponse avec température pour éviter les boucles"""
    if not prompt.strip():
        prompt = "bonjour"
    
    input_indices = [char_to_int.get(c, 0) for c in prompt]
    input_tensor = torch.tensor(input_indices, dtype=torch.long).unsqueeze(0)
    
    resultat = prompt
    hidden = None
    dernieres_phrases = []  # Pour détecter les boucles
    caractere_precedent = ""
    
    with torch.no_grad():
        for _ in range(max_len):
            out, hidden = model(input_tensor, hidden)
            
            # Appliquer la température
            logits = out[0, -1] / temperature
            probs = torch.softmax(logits, dim=-1)
            
            # Éviter la répétition excessive du même caractère
            if caractere_precedent:
                # Réduire la probabilité du caractère précédent
                idx_precedent = char_to_int.get(caractere_precedent, 0)
                if idx_precedent < len(probs):
                    probs[idx_precedent] *= 0.5
            
            prochain_char_idx = torch.multinomial(probs, 1).item()
            char = int_to_char[prochain_char_idx]
            
            resultat += char
            caractere_precedent = char
            
            # Détection de boucle (même phrase répétée)
            phrases = resultat.split('.')
            if len(phrases) > 3:
                derniere = phrases[-1].strip()
                avant_derniere = phrases[-2].strip()
                if derniere == avant_derniere and len(derniere) > 5:
                    break
            
            # Arrêt si point d'interrogation ou point final
            if char in ".!?" and len(resultat) > len(prompt) + 15:
                # Vérifier que la phrase est complète
                break
                
            input_tensor = torch.tensor([[prochain_char_idx]], dtype=torch.long)
    
    # Nettoyer la réponse
    resultat = resultat[:resultat.rfind('.')+1] if '.' in resultat else resultat
    return resultat

print("\n" + "=" * 50)
print("🤖 MOTEUR GOPI.INC ACTIF")
print("=" * 50)
print("Tape 'wiki: sujet' pour Wikipedia")
print("Tape 'quit' ou 'exit' pour quitter")
print("=" * 50)

while True:
    user_input = input("\n[Toi] > ").strip()
    
    if not user_input:
        print("[Gopu] > Tu n'as rien écrit !")
        continue
    
    if user_input.lower() in ["quit", "exit"]:
        print("[Gopu] > Au revoir ! 👋")
        break
    
    if user_input.startswith("wiki:"):
        sujet = user_input.split("wiki:", 1)[1].strip()
        if not sujet:
            print("[Wiki] > Spécifie un sujet.")
            continue
            
        print(f"🔍 Recherche Wikipedia sur : {sujet}")
        try:
            summary = wikipedia.summary(sujet, sentences=2)
            print(f"[Wiki] > {summary}")
        except:
            print("[Wiki] > Sujet non trouvé.")
    else:
        reponse = generer_reponse(user_input)
        print(f"[Gopu] > {reponse}")
