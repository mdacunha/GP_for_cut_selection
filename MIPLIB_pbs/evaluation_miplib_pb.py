import subprocess
import os
import sys
import json
import math
import time
import conf


def evaluate(function, time_limit, partition):
    if partition == "transfer":
        path = os.path.join(conf.ROOT_DIR, f"data/MIPLIB2017/Instances_unziped/")
        instance_set = os.listdir(path)
    else:
        instance_set = conf.instances_training
    print("evaluation of the function ", function, " with time limit ", time_limit)
    list_of_done = []
    performances = []

    if 'seed_' in function:  # Gp function ar in the form seed_...: seed_2 or seed_14
        seed = function[5:]
        function = conf.gp_funcs_MIPLIB_seeds[function[5:]]
        GP = True
    else:
        GP = False
    t = time.time()
    for instance in instance_set:
        MIPLILBtest = os.path.join(conf.ROOT_DIR, "MIPLIB_pbs/subprocess_execution_MIPLIB.py")
        time_out = math.ceil(int(time_limit) * 1.5)
        print("doing for ", instance)

        p = subprocess.Popen(
        ['python', MIPLILBtest, function, time_limit, instance],stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
          output, errors = p.communicate(timeout=time_out)
        except:
          p.kill()
          output, errors = p.communicate()
          print("killed")

        if '/' in output:

            # print(result.stdout)

            raw_results = output.replace("\n", "")
            raw_results = raw_results.split("/")
            # print(raw_results)
            print("done for instance: ", instance, raw_results)
            list_of_done.append(instance)
            performances.append([int(raw_results[0]), float(raw_results[1])])
        else:
            print("ERROR on instance ", instance, " with log: ", errors)
            list_of_done.append(instance)
            performances.append([0, 1e+20])
    if GP:
        saving_json_dir = os.path.join(conf.ROOT_DIR,
                                       f"simulation_outcomes/MIPLIB/time_limit_{time_limit}_function_GP_seed_{seed}_partition_{partition}.json")
    else:
        saving_json_dir = os.path.join(conf.ROOT_DIR,
                                       f"simulation_outcomes/MIPLIB/time_limit_{time_limit}_function_{function}_partition_{partition}.json")
    with open(saving_json_dir,
              "w+") as outfile:
        json.dump({'list_of_done': list_of_done, "performances": performances}, outfile)


if __name__ == "__main__":
    function = "SCIP"
    time_limit = "10"

    partition = "transfer"
    print(sys.argv)
    for i in range(1, len(sys.argv), 2):
        if sys.argv[i] == '-function':
            function = sys.argv[i + 1]
        if sys.argv[i] == '-time_limit':
            time_limit = sys.argv[i + 1]
        if sys.argv[i] == '-partition':
            partition = sys.argv[i + 1]

    evaluate(function, time_limit, partition)


