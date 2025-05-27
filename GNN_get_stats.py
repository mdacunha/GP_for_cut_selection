import yaml
import json
import os
import conf
from conf import *
import math
import numpy as np
import matplotlib.pyplot as plt

def graph(fichier1, fichier2, attr):
    def load_and_plot(fichier, label):

        if fichier.endswith('.json'):
            with open(fichier, 'r') as file:
                data = json.load(file)
            
            scores = [value[1] for value in data.values()]
            
        elif fichier.endswith('.yaml'):
            with open(fichier, 'r') as file:
                data = yaml.safe_load(file)

            scores = [entry[attr] for entry in data.values()]

        sorted_scores = np.sort(scores)
        cumulative_distribution = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores)

        # Calcul de la moyenne et de la médiane
        mean_score = np.mean(scores)
        median_score = np.median(scores)

        # Tracé du graphique
        plt.plot(sorted_scores, cumulative_distribution, label=label)

        # Ajout des lignes verticales pour la moyenne et la médiane
        plt.axvline(mean_score, color='r', linestyle='--', label=f'Mean {label} = {mean_score:.3f}')
        plt.axvline(median_score, color='g', linestyle='--', label=f'Median {label} = {median_score:.3f}')

    # Tracer les courbes pour les deux fichiers
    plt.figure(figsize=(10, 6))
    load_and_plot(fichier1, 'File 1')
    load_and_plot(fichier2, 'File 2')

    plt.xlabel('Relative GAP improvement (generated parameters)')
    plt.ylabel('Fraction of instances')
    plt.title('Cumulative Distribution Function of Scores')
    plt.legend()
    plt.grid(True)
    plt.show()

fichier1 = os.path.join(conf.ROOT_DIR, "full_network_improvements.yaml")
fichier2 = os.path.join(conf.ROOT_DIR, "Test_GP_parsimony_parameter_1.2.json")  # Remplacez par le chemin de votre second fichier
attr = "improvement"
graph(fichier1, fichier2, attr)




