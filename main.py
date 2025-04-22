import os
import shutil
import conf
from data.build_gisp_instances import *
from genetic_programming_for_node_scoring import *
from artificial_pbs.evaluation_convergence_of_GP_over_gens_articial_pbs import *
from artificial_pbs.evaluation_artificial_problems import *
from artificial_pbs.build_tables_artificial_pb_perfs import *

if __name__ == "__main__":
    # Note: The construction of the instance reuses the code by Labassi et al. on Graph Neural Networks.
    # Parameters for instance generation
    n = 50  # Number of training instances
    whichSet = 'SET2'  # Set identifier, must be set to 'SET2'
    setparam = 100  # Parameter related to "revenues"
    alphaE2 = 0.5  # Probability of building an edge

    # Graph parameters for GISP problem representation
    min_n = 60  # Minimum number of nodes in the graph
    max_n = 70  # Maximum number of nodes in the graph
    er_prob = 0.6  # Erdos-Rényi random graph parameter

    training_file = "data/gisp/train_for_jupyter"

    # Directory for training instances
    lp_dir_training = os.path.join(conf.ROOT_DIR, training_file)
    if os.path.exists(lp_dir_training):
        print(f"Cleaning directory: {lp_dir_training}")
        try:
            shutil.rmtree(lp_dir_training)  # Recursively removes a directory and all its contents
        except Exception as e:
            print(f"Error cleaning {lp_dir_training}: {e}")
    else:
        os.mkdir(lp_dir_training)

    # Generate training instances
    generate_instances(n, whichSet, setparam, alphaE2, min_n, max_n, er_prob, None, lp_dir_training, False)

    test_file = "data/gisp/test_for_jupyter"

    # Parameters for test instance generation
    n = 60  # Number of test instances, increased by 10 in case they are failures in the evaluation through all the baselines

    # Directory for test instances
    lp_dir_test = os.path.join(conf.ROOT_DIR, test_file)
    if os.path.exists(lp_dir_test):
        print(f"Cleaning directory: {lp_dir_test}")
        try:
            shutil.rmtree(lp_dir_test)  # Recursively removes a directory and all its contents
        except Exception as e:
            print(f"Error cleaning {lp_dir_test}: {e}")
    else:
        os.mkdir(lp_dir_test)

    # Generate test instances
    generate_instances(n, whichSet, setparam, alphaE2, min_n, max_n, er_prob, None, lp_dir_test, False)


    # Parameters for GP_function training
    problem = "gisp"  # Problem type
    training_folder = "train_for_jupyter"
    initial_pop = 50  # Population size for tree-based heuristics
    mate = 0.9  # Crossover rate
    mutate = 0.1  # Mutation rate
    nb_of_gen = 50  # Number of generations
    #seed = 0  # Random seed
    node_select = "BFS"  # Node selection method (BFS allows testing DFS as well)
    saving_folder = os.path.join(conf.ROOT_DIR, "outcomes_jupyter/GP_function")

    

    # Ensure the saving directory exists
    if not os.path.exists(saving_folder):
        os.makedirs(saving_folder)

    # Tournament parameters
    fitness_size = 5  # Number of individuals in the fitness tournament
    parsimony_size = 1.2  # Parameter for size-based tournament
    time_limit = 0  # Time limit (not applicable for artificial problems)
    nb_of_instances = 0  # Number of instances (not applicable for artificial problems)

    for seed in [0, 1]:
        # Construct a unique name for the run
        name = f"{problem}_pop_{initial_pop}_nb_gen{nb_of_gen}_seed_{seed}"
        # Run the GP_function training
        main_GP(
            problem=problem,
            initial_pop=initial_pop,
            mate=mate,
            mutate=mutate,
            nb_of_gen=nb_of_gen,
            seed=seed,
            node_select=node_select,
            saving_folder=saving_folder,
            name=name,
            training_folder=training_folder,
            fitness_size=fitness_size,
            parsimony_size=parsimony_size,
            time_limit=time_limit,
            nb_of_instances=nb_of_instances
        )

        # Define the folder containing simulation results (defaults to the first element in this folder)
        simulation_folder = os.path.join(conf.ROOT_DIR, "outcomes_jupyter")

        # Evaluate the convergence of GP across generations
        gp_function = convergence_of_gp_over_generations(simulation_folder,saving=False)

        gp_func_dic = {"1.2":gp_function}#1.2 is meant for the parsimony parameter
        print(gp_function)
        problem = "gisp"
        partition= "test_for_jupyter"
        evaluation_gnn_gp(problem, partition, gp_func_dic,do_gnn=False, build_set_of_instances=False,saving_folder="outcomes_jupyter")

        # Gather information from JSON files for the specified problems and partitions
        dic_info = gather_info_from_json_files(problems=["gisp"], partitions=["test_for_jupyter"], saving_folder="outcomes_jupyter")

        # Display the output results
        just_get_the_output_results(dic_info)