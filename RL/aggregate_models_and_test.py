# Aggregate models by averaging their weights.
import torch
import copy
import os
os.chdir('..')
import argparse
import conf
from RL.nn_wrapper import NeuralNetworkWrapper
from ranking import parse_filename, extract_values_from_file
from artificial_pbs.evaluation_convergence_of_GP_over_gens_articial_pbs import convergence_of_gp_over_generations
from artificial_pbs.build_tables_artificial_pb_perfs import gather_info_from_json_files, just_get_the_output_results
from artificial_pbs.evaluation_artificial_problems import evaluation_gp

def get_models(problem="gisp"):
    folder = os.path.join(conf.ROOT_DIR, 
                            "simulation_folder", 
                            "pb__" + problem,
                            "numcut__RL",
                            "not_parallel",
                            "mode_only_scores"
                        )
    models = []
    for seed in os.listdir(folder):
        fold_seed = os.path.join(folder, seed)
        model_wrapper = NeuralNetworkWrapper()
        model_wrapper.load_checkpoint(folder=fold_seed, filename="weights")
        models.append(model_wrapper.nnet)
    return models

def average_models(models, problem="gisp"):
    avg_model = copy.deepcopy(models[0])
    state_dicts = [model.state_dict() for model in models]

    averaged_dict = {}
    for key in state_dicts[0]:
        averaged_dict[key] = sum([sd[key].float() for sd in state_dicts]) / len(models)

    avg_model.load_state_dict(averaged_dict)

    folder = os.path.join(conf.ROOT_DIR, 
                            "simulation_folder", 
                            "pb__" + problem,
                            "numcut__RL"
                        )
    avg_model_wrapper = NeuralNetworkWrapper(glob_model=avg_model)  # si tu as une classe qui encapsule le modèle
    avg_model_wrapper.save_checkpoint(folder=folder, filename=f"global_{problem}_model.pth.tar")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="evaluate a GP_function candidate")
    parser.add_argument('problem', type=str, help='problem')
    parser.add_argument('test', type=bool, help='wether to test the model or not')
    args = parser.parse_args()
    models = get_models(problem=args.problem)
    average_models(models, problem=args.problem)
    if args.test:
        dossier = os.path.join(conf.ROOT_DIR, "data", args.problem, "test")
        contenu = os.listdir(dossier)
        fichiers = [f for f in contenu if os.path.isfile(os.path.join(dossier, f))]
        
        n_test_instances = len(fichiers)
        simulation_folder = os.path.join(conf.ROOT_DIR, 
                                            "simulation_folder", 
                                            "pb__" + args.problem,
                                            "numcut__RL", 
                                            "global_model_results"
                                            )
        problem_folder = os.path.join(simulation_folder, args.problem, "temp")
        os.makedirs(problem_folder, exist_ok=True)

        directory = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "logs_ind_1_heuristic")
        best_score = 0.0
        best_filename = None
        for filename in os.listdir(directory):
            if filename.endswith('.out'):
                _, _, seed, _ = parse_filename(filename)
                filepath = os.path.join(directory, filename)
                score = extract_values_from_file(filepath)
                if score > best_score:
                    best_score = score
                    best_seed = seed

        fold = os.path.join(conf.ROOT_DIR, 
                                "simulation_folder", 
                                "pb__" + args.problem,
                                "numcut__RL", 
                                "not_parallel",
                                "mode__only_scores",
                                "seed__" + best_seed, 
                                "loop__1")
        
        gp_func_dic=convergence_of_gp_over_generations(fold, saving=False, show=False)

        evaluation_gp(args.problem, n_test_instances=n_test_instances, gp_func_dic=gp_func_dic, 
                        saving_folder=simulation_folder, RL=True, inputs_type="only_scores")
        
        # Gather information from JSON files for the specified problems and partitions
        dic_info = gather_info_from_json_files(problems=[args.problem], partitions=["test"], saving_folder=simulation_folder)

        # Display the output results
        just_get_the_output_results(dic_info)
