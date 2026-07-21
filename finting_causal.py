"""
finetune_qa_phase2.py

A lancer APRES le script colab_train_locgi_v2.py (donc après avoir déjà
un modèle pré-entraîné sur corpus.txt + frenchSTS).

Objectif : ré-entraîner (fine-tuner) UNIQUEMENT sur tes lignes
"Question : ... Réponse : ..." d'origine, avec un batching qui garde
chaque exemple intact (pas de découpage à cheval sur deux lignes,
pas de mélange avec des phrases STS sans rapport).

Prérequis : avoir déjà exécuté colab_train_locgi_v2.py dans la même
session (les classes GopuTransformerLite, word_to_int, int_to_word,
vocab_size, model, device, SEQ_LEN doivent déjà exister).
"""

import re
import random
import torch
import torch.nn as nn
from safetensors.torch import save_file

# %% [1. Isoler UNIQUEMENT les lignes Question/Réponse d'origine]
with open("corpus.txt", "r", encoding="utf-8") as f:
    toutes_les_lignes = [l.strip() for l in f if l.strip()]

lignes_qa = [l for l in toutes_les_lignes if l.startswith("Question :") and "Réponse :" in l]
print(f"📚 {len(lignes_qa)} lignes Question/Réponse isolées sur {len(toutes_les_lignes)} au total")

if len(lignes_qa) < 100:
    print("⚠️ Peu de lignes QA détectées — vérifie que le format 'Question : ... Réponse : ...' est intact.")

# %% [2. Tokeniser exemple par exemple (pas de concaténation globale)]
def tokeniser(texte):
    return re.findall(r"\w+|[^\w\s]", texte.lower())

exemples = []
for ligne in lignes_qa:
    tokens = tokeniser(ligne)
    indices = [word_to_int.get(t, word_to_int["<UNK>"]) for t in tokens]
    if 2 <= len(indices) <= SEQ_LEN:
        exemples.append(indices)

print(f"📦 {len(exemples)} exemples utilisables (longueur <= {SEQ_LEN} tokens)")

# %% [3. Batching par exemple, avec padding — chaque ligne reste intacte]
PAD_ID = word_to_int["<PAD>"]

def faire_batches_qa(exemples, batch_size):
    exemples_tries = sorted(exemples, key=len)
    lots = [exemples_tries[i:i + batch_size] for i in range(0, len(exemples_tries), batch_size)]
    random.shuffle(lots)
    batches = []
    for lot in lots:
        max_len = max(len(e) for e in lot)
        if max_len < 2:
            continue
        x_batch = torch.full((len(lot), max_len - 1), PAD_ID, dtype=torch.long)
        y_batch = torch.full((len(lot), max_len - 1), PAD_ID, dtype=torch.long)
        for j, e in enumerate(lot):
            L = len(e) - 1
            x_batch[j, :L] = torch.tensor(e[:-1], dtype=torch.long)
            y_batch[j, :L] = torch.tensor(e[1:], dtype=torch.long)
        batches.append((x_batch, y_batch))
    return batches

BATCH_SIZE_QA = 16
batches_qa = faire_batches_qa(exemples, BATCH_SIZE_QA)
print(f"📚 {len(batches_qa)} batches QA (taille variable, padding par lot)")

# %% [4. Fine-tuning : LR plus faible, peu d'époques, pour ne pas tout écraser]
optimizer_ft = torch.optim.AdamW(model.parameters(), lr=5e-5)  # LR ~4x plus faible qu'en phase 1
criterion_ft = nn.CrossEntropyLoss(ignore_index=PAD_ID)

EPOCHS_FT = 15
best_loss_ft = float("inf")

for epoch in range(EPOCHS_FT):
    model.train()
    random.shuffle(batches_qa)
    total_loss = 0
    for x, y in batches_qa:
        x, y = x.to(device), y.to(device)
        optimizer_ft.zero_grad()
        out = model(x)
        loss = criterion_ft(out.reshape(-1, vocab_size), y.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer_ft.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(batches_qa)
    if avg_loss < best_loss_ft:
        best_loss_ft = avg_loss
        save_file(model.state_dict(), "gopu_poids.safetensors")

    print(f"[Fine-tune QA] Époque {epoch+1:2d}/{EPOCHS_FT} | Loss: {avg_loss:.4f} | Best: {best_loss_ft:.4f}")

print(f"✅ Fine-tuning QA terminé. Meilleure loss : {best_loss_ft:.4f}")

# %% [5. Retest]
for q in ["Quelle est la capitale de la France ?", "Qui a peint la Joconde ?",
          "Quelle est la capitale de l'Allemagne ?"]:
    print(f"[Toi] > {q}")
    print(f"[Locgi] > {generer(q)}\n")

# %% [6. Ré-upload du modèle fine-tuné]
from huggingface_hub import upload_file
upload_file(path_or_fileobj="gopu_poids.safetensors", path_in_repo="gopu_poids.safetensors", repo_id=REPO_ID)
print("📤 Modèle fine-tuné ré-uploadé sur le Hub.")

