import conf
import os
import json
sh_template = """#!/bin/bash -l
#SBATCH -c 28
#SBATCH -t 2880
#SBATCH -p batch
#SBATCH --exclusive

cd
micromamba activate a_even_better_name_than_base
    """
if __name__ == "__main__":
    nb_gen = 20
    initial_pop = 20
    saving_folder = os.path.join(os.path.dirname(__file__),
                                 'batch_jobs\GP_training\MIPLIB\\')
    sh_template = '#!/bin/bash -l\n#SBATCH -c 28\n#SBATCH -t 2880\n#SBATCH -p batch\n#SBATCH --exclusive\ncd\nmicromamba activate a_even_better_name_than_base\n'

    for seed in conf.seeds:
            nb_of_instances = 50
            time_limit = 10
            to_print=sh_template + f'python -u node-selection-method-using-gp/genetic_programming_for_node_scoring.py -problem MIPLIB -nb_of_gen {nb_gen} -initial_pop {initial_pop} -fitness_size 3 -time_limit {time_limit} -seed {seed} -nb_of_instances {nb_of_instances}'
            name = f'miplib_seed_{seed}_nb_of_instances_{nb_of_instances}_time_limit_{time_limit}'
            file_path = saving_folder + name + f".sh"
            with open(file_path
                    ,
                    "w",newline='\n') as outfile:
                outfile.write(to_print)
            os.chmod(file_path, 0o755)