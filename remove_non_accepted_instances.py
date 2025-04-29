import argparse
import os
import sys

def lister_fichiers(dossier, transformed_folder):
    """Retourne la liste des noms de fichiers dans un dossier donné."""
    if transformed_folder:
        try:
            return set(f.split("_trans")[0] + '.{}'.format('lp') for f in os.listdir(dossier) if os.path.isfile(os.path.join(dossier, f)))
        except FileNotFoundError:
            print(f"Le dossier {dossier} n'existe pas.")
            return set()

    else:
        try:
            return set(f for f in os.listdir(dossier) if os.path.isfile(os.path.join(dossier, f)))
        except FileNotFoundError:
            print(f"Le dossier {dossier} n'existe pas.")
            return set()

def supprimer_fichiers_non_communs(dossier1, dossier2, dossier_reference):
    """Supprime les fichiers des dossiers 1 et 2 qui ne sont pas dans le dossier de référence."""
    fichiers_dossier1 = lister_fichiers(dossier1)
    fichiers_dossier2 = lister_fichiers(dossier2)
    fichiers_reference = lister_fichiers(dossier_reference)

    # Supprimer les fichiers de dossier1 qui ne sont pas dans dossier_reference
    for fichier in fichiers_dossier1 - fichiers_reference:
        chemin_fichier = os.path.join(dossier1, fichier)
        os.remove(chemin_fichier)

    # Supprimer les fichiers de dossier2 qui ne sont pas dans dossier_reference
    for fichier in fichiers_dossier2 - fichiers_reference:
        chemin_fichier = os.path.join(dossier2, fichier)
        os.remove(chemin_fichier)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source_dir", type=str, help="Source directory containing the files")
    parser.add_argument("original_data_train_set", type=str, help="Source directory containing the train set files") 
    parser.add_argument("original_data_test_set", type=str, help="Source directory containing the test set files")
    args = parser.parse_args()

    dossier_reference = args.source_dir
    dossier1 = args.original_data_train_set
    dossier2 = args.original_data_test_set

    supprimer_fichiers_non_communs(dossier1, dossier2, dossier_reference)
