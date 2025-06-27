import os
import shutil
import conf
import argparse
from data.build_gisp_instances import *
from genetic_programming_for_node_scoring import *
from artificial_pbs.evaluation_convergence_of_GP_over_gens_articial_pbs import *
from artificial_pbs.evaluation_artificial_problems import *
from artificial_pbs.build_tables_artificial_pb_perfs import *

from RL.nn_wrapper import NeuralNetworkWrapper

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('problem', type=str, help="problem")
    parser.add_argument('num_cuts_per_round', type=str, help="num_cuts_per_round")
    parser.add_argument('seed', type=str, help="Seed that we choose")

    # side quests : 
    parser.add_argument('inputs_type', type=str, help="inputs_type for NN") # one of ["only_scores", "only_features", "scores_and_features"]
    parser.add_argument('GP_function_test', type=str, help="Testing folder")
    parser.add_argument('sol_path', type=str, help="Path to the solution file")
    args = parser.parse_args()

    #python main.py "gisp" "5" "0" "only_scores" None None
    
    # Parameters for GP_function training
    problem = args.problem  # Problem type
    training_folder = "train"
    more_training_folder = "more_train"
    testing_folder= "test"
    initial_pop = 127  # Population size for tree-based heuristics
    mate = 0.9  # Crossover rate
    mutate = 0.1  # Mutation rate
    nb_of_gen = 20  # Number of generations
    num_cuts_per_round = args.num_cuts_per_round
    num_cuts_per_round_by_default = "30"
    seed = args.seed  # Random seed
    inputs_type = args.inputs_type
    sol_path = args.sol_path  # Path to the solution file

    node_select = "BFS"  # Node selection method (BFS allows testing DFS as well)

    # Tournament parameters
    fitness_size = 5  # Number of individuals in the fitness tournament
    parsimony_size = 1.2  # Parameter for size-based tournament
    time_limit = 0  # Time limit (not applicable for artificial problems)
    nb_of_instances = 0  # Number of instances (not applicable for artificial problems)

    # Environment parametrisation for SCIP solving
    GNN_comparison = False
    transformed = False  # Whether to use the transformed version of the problem for comparison with GNN
    root = False
    if root:
        node_lim = 1  # Node limit for GNN comparison
    else:
        node_lim = -1

    get_scores= False  # Whether to get scores for info
    SCIP_func_test = False  # Whether to test the SCIP function
    parallel = True  # Whether to run in parallel for slurm on HPC

    heuristic = False  # DO NOT USE, set num_cuts_per_round == "heuristic" instead
    if num_cuts_per_round == "heuristic":
        heuristic = True 


    RL = False # DO NOT USE, set num_cuts_per_round == "RL" instead
    load_checkpoint = False  # Whether to load a checkpoint for first RL training
    loop = 1
    extend_training_instances = True
    if num_cuts_per_round == "RL":
        RL = True
        loop = 3

    dossier = os.path.join(conf.ROOT_DIR, "data", problem, testing_folder)
    contenu = os.listdir(dossier)
    fichiers = [f for f in contenu if os.path.isfile(os.path.join(dossier, f))]
    
    n_test_instances = len(fichiers)
    
    """########### SMALL PARAM FOR TESTING ###########
    #n_test_instances
    initial_pop=1
    nb_of_gen=0
    seed="0"
    node_lim=-1
    fitness_size=5
    loop=2
    parallel = False
    #n_test_instances = 2
    ############ SMALL PARAM FOR TESTING ###########"""

    """simulation_folder = os.path.join(conf.ROOT_DIR, "simulation_folder", "pb__" + problem + "__numcut__" + num_cuts_per_round + "__seed__" + seed)
    if not os.path.exists(simulation_folder):
        os.makedirs(simulation_folder)
    function_folder = os.path.join(simulation_folder, "GP_function")
    problem_folder = os.path.join(simulation_folder, problem)
    os.makedirs(function_folder, exist_ok=True)  # Create the problem folder if it doesn't exist
    os.makedirs(problem_folder, exist_ok=True) """

    """name = f"{problem}_pop_{initial_pop}_nb_gen{nb_of_gen}_seed_{seed}"
    """

    """if os.path.exists(f"gp_stats_{num_cuts_per_round}.txt"):
        os.remove(f"gp_stats_{num_cuts_per_round}.txt")
    if os.path.exists(f"scip_stats_{num_cuts_per_round}.txt"):
        os.remove(f"scip_stats_{num_cuts_per_round}.txt")"""

    

    for i in range(loop):
        if num_cuts_per_round == "RL":
            higher_simulation_folder = os.path.join(conf.ROOT_DIR, 
                                                    "simulation_folder", 
                                                    "pb__" + problem + "__numcut__" + num_cuts_per_round, 
                                                    "mode__" + inputs_type, 
                                                    "seed__" + seed)
            if not os.path.exists(higher_simulation_folder):
                os.makedirs(higher_simulation_folder)
            simulation_folder = os.path.join(higher_simulation_folder, "loop__" + str(i))
            if not os.path.exists(simulation_folder):
                os.makedirs(simulation_folder)
        else:
            higher_simulation_folder=""
            simulation_folder = os.path.join(conf.ROOT_DIR, "simulation_folder", 
                                             "pb__" + problem, 
                                             "numcut__" + num_cuts_per_round, 
                                             "seed__" + seed)
            if not os.path.exists(simulation_folder):
                os.makedirs(simulation_folder)
        function_folder = os.path.join(simulation_folder, "GP_function")
        problem_folder = os.path.join(simulation_folder, problem, "temp")
        os.makedirs(function_folder, exist_ok=True)  # Create the problem folder if it doesn't exist
        os.makedirs(problem_folder, exist_ok=True) 

        name = f"{problem}_pop_{initial_pop}_nb_gen{nb_of_gen}_seed_{seed}_loop_{i}"

        if num_cuts_per_round == "heuristic" or num_cuts_per_round == "RL":
            num_cuts_per_round = num_cuts_per_round_by_default          

        main_GP(
            problem=problem,
            initial_pop=initial_pop,
            mate=mate,
            mutate=mutate,
            nb_of_gen=nb_of_gen,
            seed=seed,
            node_select=node_select,
            saving_folder=function_folder,
            name=name,
            training_folder=training_folder,
            fitness_size=fitness_size,
            parsimony_size=parsimony_size,
            time_limit=time_limit,
            nb_of_instances=nb_of_instances,
            fixedcutsel=GNN_comparison,
            node_lim=node_lim,
            sol_path=sol_path,
            transformed=transformed,
            test=SCIP_func_test,
            num_cuts_per_round=num_cuts_per_round,
            parallel=parallel,
            RL=load_checkpoint,
            inputs_type=inputs_type,
            higher_simulation_folder=higher_simulation_folder,
            heuristic=heuristic
        ) 

        # Evaluate the convergence of GP across generations
        gp_function = convergence_of_gp_over_generations(simulation_folder,saving=False, show=False)

        if RL:            
            if extend_training_instances:
                training_path = [os.path.join(conf.ROOT_DIR, f"data/{problem}/{training_folder}/"), 
                                 os.path.join(conf.ROOT_DIR, f"data/{problem}/{more_training_folder}/")]
            else:
                training_path = [os.path.join(conf.ROOT_DIR, f"data/{problem}/{training_folder}/")]
                                
            testing_path = os.path.join(conf.ROOT_DIR, f"data/{problem}/{testing_folder}/")
            nnetwrapper = NeuralNetworkWrapper(training_path=training_path,
                                                testing_path=testing_path,
                                                higher_simulation_folder=higher_simulation_folder,
                                                problem=problem,
                                                cut_comp=gp_function,
                                                parameter_settings=True,
                                                saving_folder="weights",
                                                load_checkpoint=load_checkpoint,
                                                inputs_type=inputs_type,
                                                sol_path=sol_path,
                                                )
            nnetwrapper.learn()


            load_checkpoint = True
            num_cuts_per_round = "RL"
        
    if num_cuts_per_round == "heuristic" or num_cuts_per_round == "RL":
        num_cuts_per_round = num_cuts_per_round_by_default

    gp_function = gp_function if args.GP_function_test == "None" else args.GP_function_test
    gp_func_dic = {"1.2":gp_function}#1.2 is meant for the parsimony parameter "protectedDiv(getRowObjParallelism, getNNonz)"
    #print(gp_function, flush=True)

    json_path = "out.json"
    if os.path.exists(json_path):
        os.remove(json_path)
    if RL and False:
        with open(json_path, "w") as f:
            json.dump({}, f)

    parallel=True
    if parallel:
        parallelised_evaluation_gp(problem, training_folder, testing_folder, higher_simulation_folder, n_test_instances, gp_func_dic, 
                                   time_limit=time_limit, fixedcutsel=GNN_comparison, GNN_transformed=transformed, node_lim=node_lim, 
                                    sol_path=sol_path, do_gnn=False, build_set_of_instances=False,saving_folder=simulation_folder,
                                    num_cuts_per_round=num_cuts_per_round, RL=RL, inputs_type=inputs_type, heuristic=heuristic, 
                                    get_scores=get_scores)
    else:
        evaluation_gp(problem, training_folder, testing_folder, higher_simulation_folder, n_test_instances, gp_func_dic, time_limit=time_limit, 
                        fixedcutsel=GNN_comparison, GNN_transformed=transformed, node_lim=node_lim, 
                        sol_path=sol_path, do_gnn=False, build_set_of_instances=False,saving_folder=simulation_folder,
                        num_cuts_per_round=num_cuts_per_round, RL=RL, inputs_type=inputs_type, heuristic=heuristic, 
                        get_scores=get_scores)

    # Gather information from JSON files for the specified problems and partitions
    dic_info = gather_info_from_json_files(problems=[problem], partitions=[testing_folder], saving_folder=simulation_folder)

    # Display the output results
    just_get_the_output_results(dic_info)