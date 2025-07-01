import math
import random
from deap import creator, base, gp, algorithms, tools, creator

import numpy
import json
from operator import *
import subprocess
import sys
import os
import time
from functools import partial
import psutil
import gc

from conf import *
from scip_solver import perform_SCIP_instance, perform_SCIP_instances_using_a_tuned_comp_policy

# Variables globales pour paramètres d'évaluation

def evaluate(scoring_function, params):

    tree_str = str(scoring_function)
    #print(tree_str, flush=True)
    
    python_path = os.path.join(os.path.dirname(__file__), "subprocess_for_genetic.py")
    result = subprocess.run(
        ['python', python_path, tree_str, params["problem"], params["training_folder"],
         params["node_select"], str(params["time_limit"]), str(params["seed"]),
         str(params["nb_of_instances"]), str(int(params["fixedcutsel"])),
         str(params["node_lim"]), params["sol_path"], str(int(params["transformed"])), 
         str(int(params["test"])), str(params["num_cuts_per_round"]), str(int(params["RL"])),
         params["inputs_type"], params["higher_simulation_folder"], str(int(params["heuristic"]))],
        capture_output=True, text=True)
    mean_solving_time_or_gap = result.stdout.strip()

    print(tree_str + " : ", mean_solving_time_or_gap, flush=True)
    if mean_solving_time_or_gap == "" or mean_solving_time_or_gap == "nan":
        print("error: ", result.stderr)
        return 10e20,
    mean_solving_time_or_gap = mean_solving_time_or_gap.replace("\n", "")
    mean_solving_time_or_gap = float(mean_solving_time_or_gap)
    
    return mean_solving_time_or_gap,

def kill_child_processes():
    parent = psutil.Process(os.getpid())
    for child in parent.children(recursive=True):
        child.kill()

def main_GP(problem="gisp", initial_pop=50, mate=0.9, mutate=0.1,
            nb_of_gen=20, seed=None, node_select="BFS", saving_folder="simulation_outcomes/", name="",
            training_folder="Train", fitness_size=5, parsimony_size=1.2, time_limit=0, nb_of_instances=0, 
            fixedcutsel=False, semantic_algo=False, node_lim=-1, sol_path=None, transformed=False, test=False,
            num_cuts_per_round=10, parallel=False, RL=False, inputs_type="", higher_simulation_folder="",
            heuristic=False):
    
    if seed is None:
        seed = math.floor(random.random() * 10000)
    print("seed:", seed, flush=True)
    random.seed(seed)
        
    pset = gp.PrimitiveSet("main", 9)
    pset.addPrimitive(add, 2)
    pset.addPrimitive(sub, 2)
    pset.addPrimitive(mul, 2)
    pset.addPrimitive(protectedDiv, 2)  # Attention, il faut définir cette fonction ou l'importer
    
    pset.renameArguments(ARG0="getDepth")
    pset.renameArguments(ARG1="getNConss")
    pset.renameArguments(ARG2="getNVars")
    pset.renameArguments(ARG3="getNNonz")
    pset.renameArguments(ARG4="getEfficacy")
    pset.renameArguments(ARG6="getNumIntCols")
    pset.renameArguments(ARG5="getCutLPSolCutoffDistance")
    pset.renameArguments(ARG7="getObjParallelism")    
    pset.renameArguments(ARG8="getCutViolation")  

    """pset.renameArguments(ARG0="getEfficacy")
    pset.renameArguments(ARG1="getNumIntCols")
    pset.renameArguments(ARG2="getCutLPSolCutoffDistance")
    pset.renameArguments(ARG3="getObjParallelism")    
    pset.renameArguments(ARG4="getCutViolation") """ 

    """pset.renameArguments(ARG9="mean_cut_values")
    pset.renameArguments(ARG10="max_cut_values")
    pset.renameArguments(ARG11="min_cut_values")
    pset.renameArguments(ARG12="std_cut_values")
    pset.renameArguments(ARG13="mean_obj_values")
    pset.renameArguments(ARG14="max_obj_values")
    pset.renameArguments(ARG15="min_obj_values")
    pset.renameArguments(ARG16="std_obj_values")"""
    pset.addTerminal(10000000)

    if "FitnessMin" not in creator.__dict__:
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    if "Individual" not in creator.__dict__:
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin, pset=pset)

    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genGrow, pset=pset, min_=1, max_=17)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)

    # Initialisation des paramètres globaux pour évaluation
    params = {
        "problem": problem,
        "training_folder": training_folder,
        "node_select": node_select,
        "time_limit": time_limit,
        "seed": seed,
        "nb_of_instances": nb_of_instances,
        "fixedcutsel": fixedcutsel,
        "node_lim": node_lim,
        "sol_path": sol_path,
        "transformed": transformed,
        "test": test,
        "num_cuts_per_round": num_cuts_per_round,
        "RL": RL,
        "inputs_type": inputs_type,
        "heuristic": heuristic,
        "higher_simulation_folder": higher_simulation_folder
    }

    evaluate_with_params = partial(evaluate, params=params)

    toolbox.register("evaluate", evaluate_with_params)

    if parallel:
        from scoop import futures
        toolbox.register("map", futures.map)  # Utilisation de scoop
    else:
        toolbox.register("map", map)
    toolbox.register("select", tools.selDoubleTournament, fitness_size=fitness_size,
                        parsimony_size=parsimony_size, fitness_first=True)
    toolbox.register("mate", gp.cxOnePoint)

    toolbox.register("expr_mut", gp.genGrow, min_=1, max_=5)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

    pop = toolbox.population(n=initial_pop)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)

    pop, logbook = algorithms.eaSimple(pop, toolbox, mate, mutate, nb_of_gen, stats,
                                        halloffame=hof)

    print(pop, logbook, stats, hof)
    for elt in hof:
        print(str(elt))
    for elt in logbook:
        print(elt)
    if saving_folder is not None:
        try:
            os.makedirs(saving_folder)
        except FileExistsError:
            pass
        with open(os.path.join(saving_folder, f"{name}.json"), "w+") as outfile:
            json.dump([logbook, [str(elt) for elt in hof]], outfile)
    
    gc.collect()
    kill_child_processes()

if __name__ == "__main__":
    t_1 = time.time()

    # Exemple d’appel avec paramètres par défaut
    problem = "miplib"
    time_limit = 10
    partition = "train"
    initial_pop = 20
    nb_of_gen = 20
    fitness_size = 5
    seed = 1
    parsimony_size = 1.2
    node_select = "BFS"
    nb_of_instances = 50

    # Parsing simple des arguments
    for i in range(len(sys.argv)):
        if sys.argv[i] == '-problem':
            problem = sys.argv[i + 1]
        if sys.argv[i] == '-partition':
            partition = sys.argv[i + 1]
        if sys.argv[i] == '-initial_pop':
            initial_pop = int(sys.argv[i + 1])
        if sys.argv[i] == '-nb_of_gen':
            nb_of_gen = int(sys.argv[i + 1])
        if sys.argv[i] == '-fitness_size':
            fitness_size = int(sys.argv[i + 1])
        if sys.argv[i] == '-parsimony_size':
            parsimony_size = float(sys.argv[i + 1])
        if sys.argv[i] == '-node_select':
            node_select = sys.argv[i + 1]
        if sys.argv[i] == '-seed':
            seed = int(sys.argv[i + 1])
        if sys.argv[i] == '-time_limit':
            time_limit = int(sys.argv[i + 1])
        if sys.argv[i] == '-nb_of_instances':
            nb_of_instances = int(sys.argv[i + 1])

    saving_folder = os.path.join(os.path.dirname(__file__), f'simulation_outcomes/{problem}/GP_function/')
    if problem == "miplib":
        name = f"miplib_seed_{seed}_nb_of_instances_{nb_of_instances}_time_limit_{time_limit}"
    else:
        name = f"{problem}_pop_{initial_pop}_nb_gen{nb_of_gen}_seed_{seed}"

    main_GP(problem=problem, initial_pop=initial_pop, mate=0.9, mutate=0.1, nb_of_gen=nb_of_gen,
            seed=seed, node_select=node_select, saving_folder=saving_folder, name=name,
            training_folder=partition, fitness_size=fitness_size, parsimony_size=parsimony_size,
            time_limit=time_limit, nb_of_instances=nb_of_instances)

    print("total running time: ", time.time() - t_1)
    print(sys.argv)
