import subprocess
import os
import shutil
import sys
import json
from pathlib import Path
import re
from multiprocessing import Pool, cpu_count
import conf
import data.build_instances


def parallelised_evaluation_gp(problem, training_folder="train", testing_folder="test", higher_simulation_folder="", nb_of_test_instances=60, 
                               gp_func_dic="", time_limit=0, 
                            fixedcutsel=False, GNN_transformed=False, node_lim=-1, sol_path=None, 
                            do_gnn=False, build_set_of_instances=False,saving_folder="simulation_outcomes",
                            num_cuts_per_round=10, RL=False, inputs_type="", heuristic=False, get_scores=False):
    
    threads = cpu_count()#128 # number of threads per node
    
    nb_of_built_instances = 100
    json_gp_func_dic = json.dumps(gp_func_dic)
    is_ok = False
    if do_gnn:
        evaluation = {"SCIP": {}, "best_estimate_BFS": {}, "best_estimate_DFS": {}, "best_LB_BFS": {},
                      "GP_parsimony_parameter_1.2": {}, "GP_parsimony_parameter_1.4": {}, "gnn_bfs_nprimal=2": {},
                      "gnn_bfs_nprimal=100000": {}}
    else:
        """evaluation = {"SCIP": {}, "best_estimate_BFS": {}, "Test_SCIP": {}, "best_estimate_DFS": {}, "best_LB_BFS": {},
                      "GP_parsimony_parameter_1.2": {}, "GP_parsimony_parameter_1.4": {}}"""
        evaluation = {"SCIP": {}, "GP_parsimony_parameter_1.2": {}}

    done = 0

    while is_ok is False:
        print("here we go again", flush=True)

        # building instances
        if build_set_of_instances:
            try:
                shutil.rmtree(os.path.join(conf.ROOT_DIR, f"data/{problem}/{testing_folder}"))
            except:
                ''
            data.build_instances.build_new_set_of_instances(problem, testing_folder, nb_of_instances=nb_of_built_instances)

        if GNN_transformed:
            path = os.path.join(conf.ROOT_DIR, f"GNN_method/TransformedInstances/{testing_folder}")
        else:
            path = os.path.join(conf.ROOT_DIR, f"data/{problem}/{testing_folder}")
        
        is_ok = True

        instances = os.listdir(path)
        params = {
            "problem": problem,
            "training_folder": training_folder,
            "testing_folder": testing_folder,
            "higher_simulation_folder": higher_simulation_folder,
            "json_gp_func_dic": json_gp_func_dic,
            "time_limit": time_limit,
            "fixedcutsel": fixedcutsel,
            "GNN_transformed": GNN_transformed,
            "node_lim": node_lim,
            "sol_path": sol_path,
            "saving_folder": saving_folder,
            "num_cuts_per_round": num_cuts_per_round,
            "RL": RL,
            "inputs_type": inputs_type,
            "heuristic": heuristic,
            "get_scores": get_scores
        }
        instance_args = [(instance, params) for instance in instances]
        with Pool(processes=min(threads, len(instances))) as pool:
            results = pool.map(run_instance, instance_args)

        for result in results:
            if result is None:
                is_ok = False
                continue
            for (method, instance), value in result.items():
                evaluation[method][instance] = value
            done += 1

        if is_ok is True:
            if done == nb_of_test_instances:
                print("everything is solved", flush=True)
                print(evaluation)
                dir = os.path.join(conf.ROOT_DIR, f'{saving_folder}/{problem}/temp/{instance}/')
                for one_perf_file in os.listdir(dir):
                    if re.match(testing_folder, one_perf_file):
                        method = one_perf_file[len(testing_folder) + 1: len(one_perf_file) - 5]
                        new_json_dir = os.path.join(conf.ROOT_DIR,  # - 25
                                                    f'{saving_folder}/{problem}/{testing_folder}_{method}.json')
                        with open(new_json_dir,
                                    "w+") as outfile:
                            json.dump(evaluation[method], outfile)
                print("ALL GUCCI GOOD")
                return
    print("is not ok")

def run_instance(instance_and_params):
    instance, params = instance_and_params
    is_ok = True
    GP_and_SCIP = os.path.join(conf.ROOT_DIR, "artificial_pbs/subprocess_evaluation_gp_SCIPbaseline.py")
    cmd = [
        'python',
        GP_and_SCIP,
        params["problem"],
        params["training_folder"],
        params["testing_folder"],
        params["higher_simulation_folder"],
        params["json_gp_func_dic"],
        str(params["time_limit"]),
        str(int(params["fixedcutsel"])),
        str(int(params["GNN_transformed"])),
        str(params["node_lim"]),
        params["sol_path"],
        instance,
        params["saving_folder"],
        str(params["num_cuts_per_round"]),
        str(int(params["RL"])),
        params["inputs_type"],
        str(int(params["heuristic"])),
        str(int(params["get_scores"]))
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    print(f"[{instance}] stdout:", result.stdout, flush=True)
    if "It is ok for GP_function and the SCIP baseline" not in result.stdout:
        print(f"[{instance}] stderr:", result.stderr, flush=True)
        return None

    dir = os.path.join(conf.ROOT_DIR, f'{params["saving_folder"]}/{params["problem"]}/temp/{instance}/')
    one_result = {}
    for one_perf_file in os.listdir(dir):
        if re.match(params["testing_folder"], one_perf_file):
            method = one_perf_file[len(params["testing_folder"])+1: len(one_perf_file) - 5]
            one_perf_path = dir + str(one_perf_file)
            with open(one_perf_path, 'r') as openfile:
                perfs = json.load(openfile)
            one_result[(method, instance)] = perfs[list(perfs.keys())[0]]
    return one_result

def evaluation_gp(problem, training_folder="train", testing_folder="test", higher_simulation_folder="", nb_of_test_instances=60, 
                    gp_func_dic="", time_limit=0, fixedcutsel=False, GNN_transformed=False, node_lim=-1, sol_path=None, 
                    do_gnn=False, build_set_of_instances=False,saving_folder="simulation_outcomes", num_cuts_per_round=10, 
                    RL=False, inputs_type="", heuristic=False, get_scores=False, exp=0, parallel_filtering=False):
    nb_of_built_instances = 100
    json_gp_func_dic = json.dumps(gp_func_dic)
    is_ok = False
    if do_gnn:
        evaluation = {"SCIP": {}, "best_estimate_BFS": {}, "best_estimate_DFS": {}, "best_LB_BFS": {},
                      "GP_parsimony_parameter_1.2": {}, "GP_parsimony_parameter_1.4": {}, "gnn_bfs_nprimal=2": {},
                      "gnn_bfs_nprimal=100000": {}}
    else:
        """evaluation = {"SCIP": {}, "best_estimate_BFS": {}, "Test_SCIP": {}, "best_estimate_DFS": {}, "best_LB_BFS": {},
                      "GP_parsimony_parameter_1.2": {}, "GP_parsimony_parameter_1.4": {}}"""
        evaluation = {"SCIP": {}, "GP_parsimony_parameter_1.2": {}}
    done = 0
    while is_ok is False:
        print("here we go again", flush=True)



        # building instances
        if build_set_of_instances:
            try:
                shutil.rmtree(os.path.join(conf.ROOT_DIR, f"data/{problem}/{testing_folder}"))
            except:
                ''
            data.build_instances.build_new_set_of_instances(problem, testing_folder, nb_of_instances=nb_of_built_instances)

        if GNN_transformed:
            path = os.path.join(conf.ROOT_DIR, f"GNN_method/TransformedInstances/{testing_folder}")
        else:
            path = os.path.join(conf.ROOT_DIR, f"data/{problem}/{testing_folder}")
            
        for instance in os.listdir(path):
            is_ok = True
            # solving GNN
            if do_gnn:
                main_GNN = os.path.join(conf.ROOT_DIR, "artificial_pbs/subprocess_evaluation_gnn.py")
                result_gnn = subprocess.run(
                    ['python', main_GNN, problem, testing_folder, instance, saving_folder],
                    capture_output=True, text=True)
                print("result for", testing_folder, " gnn : ", result_gnn.stdout)
                if "It is ok for GNN" not in result_gnn.stdout:
                    is_ok = False
                    print("stderr: ", result_gnn.stderr)
                    try:
                        shutil.rmtree(os.path.join(conf.ROOT_DIR, f"stats/{problem}"))
                    except:
                        ''

            # solving GP_function and heuristics and SCIP
            GP_and_SCIP = os.path.join(conf.ROOT_DIR, "artificial_pbs/subprocess_evaluation_gp_SCIPbaseline.py")
            result = subprocess.run(
                ['python', GP_and_SCIP, problem, training_folder, testing_folder, higher_simulation_folder,
                                                    json_gp_func_dic, str(time_limit), 
                                                   str(int(fixedcutsel)), str(int(GNN_transformed)), str(node_lim), 
                                                   sol_path, instance, saving_folder, str(num_cuts_per_round),
                                                   str(int(RL)), inputs_type, str(int(heuristic)), str(int(get_scores)),
                                                   str(exp), str(int(parallel_filtering))],
                capture_output=True, text=True)
            print("result for", testing_folder, "GP_function and SCIP : ", result.stdout, flush=True)

            if "It is ok for GP_function and the SCIP baseline" not in result.stdout:
                is_ok = False
                print("stderr: ", result.stderr, flush=True)

            if is_ok is True:
                done+=1
                print("one is done, with total of ",done, flush=True)
                dir = os.path.join(conf.ROOT_DIR,  # - 25
                                   f'{saving_folder}/{problem}/temp/{instance}/')
                for one_perf_file in os.listdir(dir):
                    if re.match(testing_folder, one_perf_file):
                        method = one_perf_file[len(testing_folder)+1: len(one_perf_file) - 5]
                        one_perf_path = dir + str(one_perf_file)
                        # print(instance)
                        with open(
                                one_perf_path,
                                'r') as openfile:
                            perfs = json.load(openfile)
                        evaluation[method][instance] = perfs[list(perfs.keys())[0]]

                if done == nb_of_test_instances:
                    print("everything is solved", flush=True)
                    print(evaluation)
                    for one_perf_file in os.listdir(dir):
                        if re.match(testing_folder, one_perf_file):
                            method = one_perf_file[len(testing_folder) + 1: len(one_perf_file) - 5]
                            new_json_dir = os.path.join(conf.ROOT_DIR,  # - 25
                                                        f'{saving_folder}/{problem}/{testing_folder}_{method}.json')
                            
                            with open(new_json_dir,
                                        "w+") as outfile:
                                json.dump(evaluation[method], outfile)
                    print("ALL GUCCI GOOD")
                    return
    print("is not ok")

if __name__ == "__main__":

    problem = "wpms"
    testing_folder = 'test'
    print(sys.argv)
    for i in range(1, len(sys.argv), 2):
        if sys.argv[i] == '-problem':
            problem = sys.argv[i + 1]
        if sys.argv[i] == '-testing_folder':
            testing_folder = sys.argv[i + 1]
    evaluation_gp(problem, testing_folder, conf.gp_funcs_artificial_problems[problem])
