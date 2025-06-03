#set files ranking

import os
import re
import argparse

def extract_values_from_file(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()[-10:]  # on regarde les 10 dernières lignes pour être sûr

    gp_time = None
    scip_time = None

    for line in lines:
        if 'geometric mean solving time for set test for method GP_parsimony_parameter_1.2' in line:
            match = re.search(r'is \$([\d.]+)', line)
            if match:
                gp_time = float(match.group(1))
        elif 'geometric mean solving time for set test for method SCIP' in line:
            match = re.search(r'is \$([\d.]+)', line)
            if match:
                scip_time = float(match.group(1))

    if gp_time is not None and scip_time is not None:
        return scip_time - gp_time
    else:
        return None

def main(directory):
    differences = []

    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        diff = extract_values_from_file(filepath)
        if diff is not None:
            differences.append((filename, diff))

    # Tri par différence décroissante
    sorted_files = sorted(differences, key=lambda x: x[1], reverse=True)

    # Affichage des fichiers triés
    print("Classement des fichiers (par écart décroissant SCIP - GP_parsimony):")
    for filename, diff in sorted_files:
        print(f"{filename}: {diff:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classe les fichiers .out par écart de temps entre SCIP et GP_parsimony.")
    parser.add_argument("directory", help="Chemin du dossier contenant les fichiers .out")
    args = parser.parse_args()

    main(args.directory)

#python classement_solver.py /chemin/vers/ton/dossier
