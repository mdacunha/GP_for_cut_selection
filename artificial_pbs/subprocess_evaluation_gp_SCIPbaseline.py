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


def evaluate_a_function_and_store_it(problem, function, performance_folder, saving_folder,
                                     partition, parameter_settings=False,func_name=None,instances=[]):
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

    perfs = {}
    if instances == []:
        path = os.path.join(conf.ROOT_DIR, performance_folder)
        instances = os.listdir(path)

    for instance in instances:
        if instance[len(instance) - 3:] == ".lp":
            path = os.path.join(conf.ROOT_DIR, performance_folder)
            instance_path = path +'/'+ str(instance)
            nb_nodes,time = perform_SCIP_instance(instance_path, cut_comp=comp_policy, node_select=sel_policy,
                                                   parameter_settings=parameter_settings)

            perfs[instance[:len(instance) - 3]] = [nb_nodes, time]

    if func_name is None:
        new_json_dir = os.path.join(os.path.abspath('')[:len(os.path.abspath(''))],  # - 25
                                    f'{saving_folder}/{problem}/{partition}_{function}.json')
        print("perfs are done for the function ", function)
    else:
        new_json_dir = os.path.join(os.path.abspath('')[:len(os.path.abspath(''))],  # - 25
                                    f'{saving_folder}/{problem}/{partition}_{func_name}.json')
        print("perfs are done for the function ", func_name)

    with open(new_json_dir,
              "w+") as outfile:
        json.dump(perfs, outfile)


def evaluate_the_GP_heuristics_and_SCIP_functions(problem, partition="test", GP_dics=None,
                                                  parameter_settings=False,instances=[], saving_folder="simulation_outcomes"):
    folder = f"data/{problem}/{partition}/"
    if GP_dics is not None:
        for key in GP_dics.keys():
            evaluate_a_function_and_store_it(problem, GP_dics[key], folder, saving_folder, partition,
                                             parameter_settings=parameter_settings,func_name="GP_parsimony_parameter_"+str(key), instances=instances)


    functions = ["SCIP"]#, "GNN"]#"best_estimate_BFS","best_estimate_DFS","best_LB_BFS"]
    for function in functions:
        evaluate_a_function_and_store_it(problem, function, folder, saving_folder, partition,
                                         parameter_settings=parameter_settings,instances=instances)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('problem', type=str, help='problem')
    parser.add_argument('partition', type=str, help='partition')
    parser.add_argument('json_gp_func_dic', type=str, help='json_gp_func_dic')
    parser.add_argument('instance', type=str, help='instance')
    parser.add_argument('saving_folder', type=str, help='saving_folder')
    args = parser.parse_args()
    problem = args.problem
    partition = args.partition

    gp_func_dic = ast.literal_eval(args.json_gp_func_dic)
    instance = args.instance


    evaluate_the_GP_heuristics_and_SCIP_functions(problem, partition=partition, GP_dics=gp_func_dic, parameter_settings=True,instances=[instance], saving_folder=args.saving_folder)
    print("It is ok for GP_function and the SCIP baseline")
