import os
import random
import shutil
import argparse
from utilities import is_dir

def select_random_files(mode, n, original_data, copy_test_set, source_dir, train_folder, test_folder):
    # Vérifier si le répertoire source existe
    if not os.path.exists(source_dir):
        print(f"Le répertoire source '{source_dir}' n'existe pas.")
        return
    
    if os.path.exists(test_folder):
        shutil.rmtree(test_folder)  # Supprimer le répertoire de destination s'il existe déjà
    if os.path.exists(train_folder):
        shutil.rmtree(train_folder)

    # Obtenir la liste de tous les fichiers dans le répertoire source
    all_files = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]

    if mode == "test":
        # Vérifier si le nombre de fichiers demandé est supérieur au nombre de fichiers disponibles
        if n > len(all_files):
            print(f"Le répertoire source contient seulement {len(all_files)} fichiers. Impossible de sélectionner {n} fichiers.")
            return

        os.makedirs(test_folder)
            
        # Sélectionner aléatoirement n fichiers
        selected_files = random.sample(all_files, n)

        # Copier les fichiers sélectionnés dans le répertoire de destination
        for file_name in selected_files:
            src_file = os.path.join(source_dir, file_name)
            dest_file = os.path.join(test_folder, file_name)
            shutil.copy2(src_file, dest_file)
    
    elif mode == "train+test":
        if copy_test_set:
            os.makedirs(train_folder)
            os.makedirs(test_folder)

            # Copier le test set de GP dans le répertoire de destination
            for file_name in all_files:
                file = file_name.split("__trans")[0]
                src_file = os.path.join(original_data, file + '.{}'.format('lp'))
                or_file_name = os.path.join(source_dir, file_name)
                if os.path.exists(src_file):
                    dest_file = os.path.join(test_folder, file_name)
                    shutil.copy2(or_file_name, dest_file)
                else:
                    dest_file = os.path.join(train_folder, file_name)
                    shutil.copy2(or_file_name, dest_file)
        else:
            # Sélectionner aléatoirement n fichiers pour le test
            n = int(len(all_files) * (n / 100))
            test_files = random.sample(all_files, n)
            train_files = [f for f in all_files if f not in test_files]

            os.makedirs(train_folder)
            os.makedirs(test_folder)

            for file_name in test_files:
                src_file = os.path.join(source_dir, file_name)
                dest_file = os.path.join(test_folder, file_name)
                shutil.copy2(src_file, dest_file)

            for file_name in train_files:
                src_file = os.path.join(source_dir, file_name)
                dest_file = os.path.join(train_folder, file_name)
                shutil.copy2(src_file, dest_file)

    print("Generated test and train sets successfully.", flush=True)

# Exemple d'utilisation
# source_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "TransformedInstances") 
# n = 20  # Remplacez par le nombre de fichiers (ex : 2) que vous souhaitez sélectionner ou le pourcentage (ex : 20)
# mode = "train+test"  # Remplacez par le mode souhaité ("test" ou "train+test")

if __name__ == "__main__":
    """
    If we only want test folder, n=number of files to test.
    If we want train and test data, n=% of files to test, and the rest will be used for training.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('original_data_test_set', type=is_dir, help="Source directory containing data files to copy the split already done for GP")
    parser.add_argument('source_dir', type=is_dir, help="Source directory containing files")
    parser.add_argument('percentage_or_number_for_test_set', type=int, help="Number of files to select for test set or percentage of files to select")
    parser.add_argument('mode', type=str, choices=['test', 'train+test'], help="Mode: 'test' or 'train+test'")
    parser.add_argument('copy_test_set', type=bool, help="True if you want to copy the test set of GP, False otherwise")
    args = parser.parse_args()
    original_data = args.original_data_test_set
    source_directory = args.source_dir #os.path.join(os.path.dirname(os.path.abspath(__file__)), "TransformedInstances/all") 
    n = args.percentage_or_number_for_test_set  # Remplacez par le nombre de fichiers (ex : 2) que vous souhaitez sélectionner ou le pourcentage (ex : 20) si pas de copie des données de test de GP
    mode = args.mode  # Remplacez par le mode souhaité ("test" ou "train+test")
    copy_test_set = args.copy_test_set  # Remplacez par True ou False selon que vous souhaitez copier le test set de GP ou non
    select_random_files(mode, n, original_data, copy_test_set, source_dir=os.path.join(source_directory, "All"), train_folder=os.path.join(source_directory, "Train"), test_folder=os.path.join(source_directory, "Test"))
