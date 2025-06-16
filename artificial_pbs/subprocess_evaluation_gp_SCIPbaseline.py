import json
import statistics
import matplotlib.pyplot as plt
import re
import argparse
import ast
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import conf
from scip_solver import perform_SCIP_instance
from conf import *

from RL.nn_wrapper import NeuralNetworkWrapper


def evaluate_a_function_and_store_it(problem, function, performance_folder, saving_folder,
                                     training_folder, testing_folder, parameter_settings=False, time_limit=0, 
                                     fixedcutsel=False, node_lim=-1, sol_path=None, func_name=None,
                                     instances=[], num_cuts_per_round=10, RL=False, heuristic=False,
                                     get_scores=False):
    
    if function == "best_estimate_BFS":
        comp_policy = "estimate"
        sel_policy = 'BFS'
    elif function == "best_estimate_DFS":
        comp_policy = "estimate"
        sel_policy = 'DFS'
    elif function == "best_LB_BFS":
        comp_policy = "LB"
        sel_policy = 'BFS'

    else:
        comp_policy = function
        sel_policy = "BFS"

    if Rl:
        training_path=os.path.join(conf.ROOT_DIR, f"data/{problem}/{training_folder}/")
        testing_path=os.path.join(conf.ROOT_DIR, f"data/{problem}/{testing_folder}/")
        nnetwrapper = NeuralNetworkWrapper(training_path=training_path,
                                            testing_path=testing_path,
                                            simulation_folder=saving_folder,
                                            problem=problem,
                                            cut_comp=comp_policy,
                                            parameter_settings=True,
                                            saving_folder="weights",
                                            load_checkpoint=True,
                                            sol_path=sol_path
                                            )
        neural_net = os.path.join(saving_folder, "weights")
        nnetwrapper.load_checkpoint(neural_net)
    else:
        nnetwrapper = None

    perfs = {}
    path = os.path.join(conf.ROOT_DIR, performance_folder)
    if instances == []:
        instances = os.listdir(path)

    for instance in instances:
        if instance[len(instance) - 3:] == ".lp" or instance[len(instance) - 4:] == ".mps":
            instance_path = path +'/'+ str(instance)
            nb_nodes,time = perform_SCIP_instance(instance_path, cut_comp=comp_policy, node_select=sel_policy,
                                                   parameter_settings=parameter_settings, time_limit=time_limit, 
                                                   fixedcutsel=fixedcutsel, node_lim=node_lim, sol_path=sol_path, 
                                                   is_Test=True, num_cuts_per_round=num_cuts_per_round, 
                                                   RL=RL, nnetwrapper=nnetwrapper, heuristic=heuristic, get_scores=get_scores)

            perfs[instance[:len(instance) - 3]] = [nb_nodes, time]

    if func_name is None:
        new_json_dir = os.path.join(os.path.abspath('')[:len(os.path.abspath(''))],  # - 25
                                    f'{saving_folder}/{problem}/{testing_folder}_{function}.json')
        print("perfs are done for the function ", function, flush=True)
    else:
        new_json_dir = os.path.join(os.path.abspath('')[:len(os.path.abspath(''))],  # - 25
                                    f'{saving_folder}/{problem}/{testing_folder}_{func_name}.json')
        print("perfs are done for the function ", func_name, flush=True)

    with open(new_json_dir,
              "w+") as outfile:
        json.dump(perfs, outfile)


def evaluate_the_GP_heuristics_and_SCIP_functions(problem, training_folder="train", testing_folder="test", GP_dics=None,
                                                  parameter_settings=False, time_limit=0, 
                                                   fixedcutsel=False, GNN_transformed=False, 
                                                   node_lim=-1, sol_path=None, instances=[], 
                                                   saving_folder="simulation_outcomes",
                                                   num_cuts_per_round=10, RL=False, heuristic=False,
                                                   get_scores=False):
    if GNN_transformed:
        folder = f"GNN_method/TransformedInstances/{testing_folder}"
    else:
        folder = f"data/{problem}/{testing_folder}"
    if GP_dics is not None:
        for key in GP_dics.keys():
            evaluate_a_function_and_store_it(problem, GP_dics[key], folder, saving_folder, training_folder, testing_folder,
                                             parameter_settings=parameter_settings, time_limit=time_limit, 
                                                   fixedcutsel=fixedcutsel, node_lim=node_lim, sol_path=sol_path, 
                                                   func_name="GP_parsimony_parameter_"+str(key), instances=instances,
                                                   num_cuts_per_round=num_cuts_per_round, RL=RL, heuristic=heuristic,
                                                   get_scores=get_scores)


    functions = ["SCIP"]#"best_estimate_BFS","best_estimate_DFS","best_LB_BFS"]
    for function in functions:
        evaluate_a_function_and_store_it(problem, function, folder, saving_folder, testing_folder,
                                         parameter_settings=parameter_settings, time_limit=time_limit, 
                                         sol_path=sol_path, instances=instances, num_cuts_per_round=num_cuts_per_round)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('problem', type=str, help='problem')
    parser.add_argument('training_folder', type=str, help='testing_folder')
    parser.add_argument('testing_folder', type=str, help='testing_folder')
    parser.add_argument('json_gp_func_dic', type=str, help='json_gp_func_dic')
    parser.add_argument('time_limit', type=int, help='time limit for the solver')
    parser.add_argument('fixedcutsel', type=int, help='0 ou 1, fixedcutsel')
    parser.add_argument('GNN_transformed', type=int, help='0 ou 1, tranformed')
    parser.add_argument('node_lim', type=int, help='node limit for the solver')
    parser.add_argument('sol_path', type=str, help='solution path for the solver')
    parser.add_argument('instance', type=str, help='instance')
    parser.add_argument('saving_folder', type=str, help='saving_folder')
    parser.add_argument('num_cuts_per_round', type=int, help='num_cuts_per_round')
    parser.add_argument('RL', type=int, help='RL')
    parser.add_argument('heuristic', type=int, help='heuristic')
    parser.add_argument('get_scores', type=int, help='get_scores')
    args = parser.parse_args()
    problem = args.problem
    training_folder=args.training_folder
    testing_folder = args.testing_folder
    fixedcutsel = bool(args.fixedcutsel)
    GNN_transformed = bool(args.GNN_transformed)
    Rl = bool(args.RL)
    heuristic = bool(args.heuristic)

    gp_func_dic = ast.literal_eval(args.json_gp_func_dic)
    instance = args.instance

    evaluate_the_GP_heuristics_and_SCIP_functions(problem, training_folder=training_folder, testing_folder=testing_folder, GP_dics=gp_func_dic, parameter_settings=True, 
                                                  time_limit=args.time_limit, fixedcutsel=fixedcutsel, 
                                                  GNN_transformed=GNN_transformed, node_lim=args.node_lim, 
                                                  sol_path=args.sol_path, instances=[instance], saving_folder=args.saving_folder, 
                                                  num_cuts_per_round=args.num_cuts_per_round, RL=Rl, heuristic=heuristic,
                                                  get_scores=args.get_scores)
    print("It is ok for GP_function and the SCIP baseline")
