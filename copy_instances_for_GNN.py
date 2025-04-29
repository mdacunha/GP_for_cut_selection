import os
import shutil
import argparse

# Fonction pour copier les fichiers d'un dossier source vers le dossier de destination
def copier_fichiers(dossier_source, dossier_destination):
    for fichier in os.listdir(dossier_source):
        chemin_fichier_source = os.path.join(dossier_source, fichier)
        if os.path.isfile(chemin_fichier_source):
            shutil.copy(chemin_fichier_source, dossier_destination)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source_dir_1", type=str, help="Path to the first source directory")
    parser.add_argument("source_dir_2", type=str, help="Path to the second source directory")
    parser.add_argument("destination_dir", type=str, help="Path to the destination directory")
    args = parser.parse_args()

    # Chemins des dossiers sources
    dossier_source_1 = args.source_dir_1  # 'chemin/vers/dossier1'
    dossier_source_2 = args.source_dir_2  # 'chemin/vers/dossier2'

    # Chemin du nouveau dossier de destination
    dossier_destination = args.destination_dir  # 'chemin/vers/nouveau_dossier'

    # Créer le dossier de destination s'il n'existe pas
    os.makedirs(dossier_destination, exist_ok=True)

    # Copier les fichiers des deux dossiers sources vers le dossier de destination
    copier_fichiers(dossier_source_1, dossier_destination)
    copier_fichiers(dossier_source_2, dossier_destination)

    print("Data successfully copied", flush=True)
