import json
import os
import argparse
import ast
import time

from scip_solver import perform_SCIP_instance


def evaluate_a_function_and_store_it(function, time_limit, instance):

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
    folder = f'data/MIPLIB2017/Instances_unziped/'
    instance_path = os.path.dirname(__file__) + "/" + folder + str(instance)
    nb_nodes, time = perform_SCIP_instance(instance_path, node_comp=comp_policy, node_select=sel_policy,
                                           parameter_settings=True, time_limit=int(time_limit))
    print(nb_nodes, "/", time)
    return nb_nodes,time


if __name__ == "__main__":
    problem = "wpms"
    partition = "test"
    gp_func_dic = "sub(getNVars,getLowerbound)"

    parser = argparse.ArgumentParser()
    parser.add_argument('function', type=str, help='function')
    parser.add_argument('time_limit', type=str, help='time_limit')
    parser.add_argument('instance', type=str, help='instance')
    args = parser.parse_args()
    function = args.function
    time_limit = args.time_limit
    instance = args.instance
    evaluate_a_function_and_store_it(function, time_limit, instance)











