import os
import shutil
import conf

dossiers_a_supprimer = [
    "data/gisp/train_for_jupyter",
    "data/gisp/test_for_jupyter"
]

for dossier in dossiers_a_supprimer:
    doss = os.path.join(conf.ROOT_DIR, dossier)
    if os.path.exists(doss):
        print(f"Suppression du dossier : {dossier}")
        shutil.rmtree(doss)
    else:
        print(f"Dossier introuvable : {dossier}")

racine = os.getcwd()

for root, dirs, _ in os.walk(racine):
    for d in dirs:
        if d == '__pycache__':
            chemin_pycache = os.path.join(root, d)
            print(f"Suppression de __pycache__ : {chemin_pycache}")
            shutil.rmtree(chemin_pycache)
