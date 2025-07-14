import argparse
import os
import random

import scip_solver
import conf

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="evaluate a GP_function candidate")
    parser.add_argument('comp_policy', type=str, help='comp_policy')
    parser.add_argument('problem', type=str, help='problem')
    parser.add_argument('training_folder', type=str, help='training_folder')
    parser.add_argument('node_select', type=str, help='node_select')
    parser.add_argument('time_limit', type=int, help='time_limit')
    parser.add_argument('seed', type=int, help='seed')
    parser.add_argument('nb_of_instances', type=int, help='nb_of_instances')
    parser.add_argument('fixedcutsel', type=int, help='0 ou 1, fixedcutsel mode')
    parser.add_argument('node_lim', type=int, help='node_lim')
    parser.add_argument('sol_path', type=str, help='sol_path')
    parser.add_argument('transformed', type=int, help='0 ou 1, tranformed mode')
    parser.add_argument('test', type=int, help='test mode for SCIP')
    parser.add_argument('num_cuts_per_round', type=int, help='number of cuts per round')
    parser.add_argument('RL', type=int, help='0 ou 1, RL mode')
    parser.add_argument('inputs_type', type=str, help='inputs_type')
    parser.add_argument('higher_simulation_folder', type=str, help='higher_simulation_folder')
    parser.add_argument('heuristic', type=int, help='0 ou 1, heuristic mode')
    parser.add_argument('exp', type=int, help='experiment number for heuristic mode')
    args = parser.parse_args()
    fixedcutsel = bool(args.fixedcutsel)
    transformed = bool(args.transformed)
    test = bool(args.test)
    RL = bool(args.RL)
    heuristic = bool(args.heuristic)
    if args.problem in ["gisp", "wpsm", "fcmcnf"]:
        if transformed:
            lp_dir = os.path.join(os.path.dirname(__file__), f"GNN_method/TransformedInstances/{args.training_folder}/")
        else:
            lp_dir = os.path.join(os.path.dirname(__file__), f"data/{args.problem}/{args.training_folder}/")
        meannnodes, mean_val = scip_solver.perform_SCIP_instances_using_a_tuned_comp_policy(instances_folder=lp_dir,
            cut_comp=args.comp_policy, node_select=args.node_select, parameter_settings=True, fixedcutsel=fixedcutsel, 
            node_lim=args.node_lim, time_limit=args.time_limit, sol_path=args.sol_path, test=test, 
            num_cuts_per_round=args.num_cuts_per_round, RL=RL, inputs_type=args.inputs_type, 
            higher_simulation_folder=args.higher_simulation_folder, heuristic=heuristic, exp=args.exp)
        print(mean_val)
    else:
        random.seed(args.seed)

        lp_dir = os.path.join(os.path.dirname(__file__), f"data/MIPLIB2017/Instances_unziped")
        training_indexes = random.sample(range(0, len(conf.instances_training)), args.nb_of_instances)
        instances = [conf.instances_training[i] for i in training_indexes]
        instances_indexes = []
        index = 0
        for elt in os.listdir(lp_dir):
            if elt in instances:
                instances_indexes.append(index)
            index += 1
        meannnodes, mean_val = scip_solver.perform_SCIP_instances_using_a_tuned_comp_policy(
            instances_folder=lp_dir,
            cut_comp=args.comp_policy, node_select=args.node_select, parameter_settings=True, fixedcutsel=fixedcutsel, 
            node_lim=args.node_lim, time_limit=args.time_limit, sol_path=args.sol_path)
        print(mean_val)
