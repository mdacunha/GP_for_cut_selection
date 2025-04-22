import conf
import os
import json
import math
sh_template = """#!/bin/bash -l
#SBATCH -c 18
#SBATCH -t 2880
#SBATCH -p batch
#SBATCH --exclusive

cd
micromamba activate a_even_better_name_than_base
    """

def building_all_batches():
    saving_folder = os.path.join(conf.ROOT_DIR,
                                 'batch_jobs\evaluation_of_all_methods\MIPLIB\\')
    for partition, nb_of_instances in zip(["test", "transfer"], [166, 758]):
        for time_limit in conf.time_limits:
            execution_time = math.ceil(nb_of_instances * time_limit * 1.5 / 60)
            sh_template = f'#!/bin/bash -l\n#SBATCH -c 28\n#SBATCH -t {execution_time}\n#SBATCH -p batch\n#SBATCH --exclusive\ncd\nmicromamba activate a_even_better_name_than_base\n'
            for seed in conf.gp_funcs_MIPLIB_seeds.keys():
                to_print = sh_template + f'python -u node-selection-method-using-gp/evaluation_miplib_pb.py -function seed_{seed} -time_limit {time_limit} -partition {partition}'
                name = f'gp_eval_miplib_seed_{seed}_time_limit_{time_limit}_partition_{partition}'
                file_path = saving_folder + name + f".sh"
                with open(file_path,"w", newline='\n') as outfile:
                    outfile.write(to_print)
                os.chmod(file_path, 0o755)
            to_print = sh_template + f'python -u node-selection-method-using-gp/evaluation_miplib_pb.py -function SCIP -time_limit {time_limit} -partition {partition}'
            name = f'gp_eval_miplib_SCIP_time_limit_{time_limit}_partition_{partition}'
            file_path = saving_folder + name + f".sh"
            with open(file_path
                    ,
                      "w", newline='\n') as outfile:
                outfile.write(to_print)
            os.chmod(file_path, 0o755)



def build_batch_jobs_handcrafted_heuristics():
    saving_folder = os.path.join(conf.ROOT_DIR,
                                 'batch_jobs\evaluation_of_all_methods\MIPLIB\\')

    for partition, nb_of_instances in zip(["transfer"], [758]):
        for time_limit in [100,110,120,130,140]:
        #for time_limit in conf.time_limits:
            if time_limit not in conf.time_limits_for_paper:
                execution_time = math.ceil(nb_of_instances * time_limit * 1.5 / 60)
                sh_template = f'#!/bin/bash -l\n#SBATCH -c 28\n#SBATCH -t {execution_time}\n#SBATCH -p batch\n#SBATCH --exclusive\ncd\nmicromamba activate a_even_better_name_than_base\n'
                for function in ["best_estimate_BFS","best_estimate_DFS"]:#,"best_LB_BFS"
                    to_print = sh_template + f'python -u node-selection-method-using-gp/evaluation_miplib_pb.py -function {function} -time_limit {time_limit} -partition {partition}'
                    name = f'gp_eval_miplib_{function}_time_limit_{time_limit}_partition_{partition}'
                    file_path = saving_folder + name + f".sh"
                    with open(file_path, "w", newline='\n') as outfile:
                        outfile.write(to_print)
                    os.chmod(file_path, 0o755)


if __name__ == "__main__":
    building_all_batches()
    build_batch_jobs_handcrafted_heuristics()