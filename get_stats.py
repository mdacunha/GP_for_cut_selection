import yaml
import os
import conf
from conf import *


def moyenne_des_scores(fichier):
    with open(fichier, 'r') as file:
        data = yaml.safe_load(file)

    gaps = [entry['gap'] for entry in data.values()]

    moyenne = shifted_geo_mean(gaps)
    return moyenne

# Exemple d'utilisation
fichier = os.path.join(conf.ROOT_DIR, "GNN_method", "ResultsGCNN", "full_network_improvements.yaml")
moyenne = moyenne_des_scores(fichier)
print(f"La moyenne des scores est : {moyenne}")
