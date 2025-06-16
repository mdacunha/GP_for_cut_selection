import os
import re
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import PercentFormatter
import numpy as np

def extract_values_from_file(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()[-10:]

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
        return (scip_time - gp_time) / scip_time
    else:
        return None

def parse_filename(filename):
    # exemple : job_MIPLIB2017_40_1.out
    match = re.match(r'job_([^_]+)_([0-9]+)_([0-9]+)\.out', filename)
    if match:
        problem = match.group(1)
        nb_cuts = int(match.group(2)) if match.group(2).isdigit() else 10000000
        seed = int(match.group(3))
        return problem, nb_cuts, seed
    else:
        return None, None, None

def main(directory):
    data = []
    full=True

    for filename in os.listdir(directory):
        if filename.endswith('.out'):
            problem, nb_cuts, seed = parse_filename(filename)
            if problem is not None:
                filepath = os.path.join(directory, filename)
                score = extract_values_from_file(filepath)
                if score is not None:
                    data.append({
                        "problem": problem,
                        "nb_cuts": nb_cuts,
                        "seed": seed,
                        "score": score
                    })
                else:
                    full=False

    df = pd.DataFrame(data)
    df_sorted = df.sort_values(by=["score"], ascending=False).reset_index(drop=True)
    return df_sorted, full

def find(df, pb, nb_cuts, seed):

    return df[(df['problem'] == pb) & (df['nb_cuts'] == nb_cuts) & (df['seed'] == seed)]

def plot_evolution_with_point(dfs, folder, lowbound=-np.inf, mean=True, median=False, points=True, show=False, save=False, compare=False):

    plt.figure(figsize=(10, 6))
    for df in dfs:
        df = df[df["score"] >= lowbound]

        problems = sorted(df["problem"].unique())
        palette = sns.color_palette("Set1", n_colors=len(problems))
        color_map = dict(zip(problems, palette))

        cut_values = sorted(df["nb_cuts"].unique())
        
        # Définir un petit offset horizontal pour chaque problème
        offsets = np.linspace(-0.8, 0.8, len(problems))

        # Courbes moyennes
        for problem in problems:
            if mean:
                mean_df = df[df["problem"] == problem].groupby("nb_cuts")["score"].mean().reset_index()
                plt.plot(mean_df["nb_cuts"], mean_df["score"] * 100, marker='o',
                        label=f"Moyenne - {problem}", color=color_map[problem])
            if median:
                median_df = df[df["problem"] == problem].groupby("nb_cuts")["score"].median().reset_index()
                plt.plot(median_df['nb_cuts'], median_df['score'] * 100, marker='o',
                        label=f"Mean - {problem}", color=color_map[problem], linestyle='--', linewidth=1.8)

        # Points individuels avec décalage horizontal
        for i, problem in enumerate(problems):
            if points:
                sub_df = df[df["problem"] == problem].copy()
                sub_df["nb_cuts_jittered"] = sub_df["nb_cuts"] + offsets[i]
                plt.scatter(sub_df["nb_cuts_jittered"], sub_df["score"] * 100,
                            label=f"Points - {problem}", color=color_map[problem], alpha=0.6, s=40, marker='x')

    plt.xlabel("Nombre de coupes")
    plt.ylabel("Écart relatif (%)")
    plt.title("Évolution de l'écart relatif selon le nombre de coupes")
    plt.grid(True)
    plt.legend(title="Problèmes", loc="best")
    plt.gca().yaxis.set_major_formatter(PercentFormatter(decimals=0))
    plt.xticks(cut_values)
    plt.tight_layout()
    if save:
        if lowbound == -np.inf:
            plt.savefig(os.path.join(folder, "mean_evolution_with_points.png"))
        else:
            plt.savefig(os.path.join(folder, f"mean_evolution_with_points_{lowbound}.png"))
    if show:
        plt.show()
    else:
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyse les fichiers .out et trace l'évolution de l'écart relatif.")
    parser.add_argument("directories", nargs='+', help="Chemin du dossier contenant les fichiers .out")
    args = parser.parse_args()

    directories = [os.path.abspath(directory) for directory in args.directories]

    if len(directories) == 1:
        compare = False
        directory_ = args.directories[0]
        directory = os.path.abspath(directory_)

        df_result = []
        df, full = main(directory)
        df_result.append(df)
        
        folder = f"{directory_[3:]}_results"
        folder = os.path.join(directory, folder)
        if not os.path.exists(folder):
            os.makedirs(folder)

        table_name = "full_results.csv" if full else "results.csv"
        df.to_csv(os.path.join(folder, table_name), index=False)
    else:
        compare = True
        df_result = []
        for directory_ in directories:
            directory = os.path.abspath(directory_)
            df, full = main(directory)

            df_result.append(df)


    # Use :

    plot_evolution_with_point(df_result, folder, 
                              lowbound=-np.inf, mean=True, median=False, points=True, show=True, save=True, compare=compare)

    #print(df_result.to_string(index=False))

    #print(find(df_result, 'gisp', 5, 0))







#python ranking.py ../logs_2
