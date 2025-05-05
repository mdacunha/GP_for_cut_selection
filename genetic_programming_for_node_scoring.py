import math

import random
import statistics

from deap import creator, base, gp, algorithms, tools
import numpy
import json
from operator import *
import subprocess
import sys
import matplotlib.pyplot as plt
import os
import re
import argparse
import time

from conf import *
from scip_solver import perform_SCIP_instance, perform_SCIP_instances_using_a_tuned_comp_policy

toolbox = base.Toolbox()


def main_GP(problem="gisp", initial_pop=50, mate=0.9, mutate=0.1,
            nb_of_gen=20, seed=None, node_select="BFS", saving_folder="simulation_outcomes/", name="",
            training_folder="Train", fitness_size=5, parsimony_size=1.2, time_limit=0, nb_of_instances=0, 
            fixedcutsel=False, node_lim=-1, sol_path=None, transformed=False):
    tree_scores = {}
    if seed is None:
        seed = math.floor(random.random() * 10000)
    print("seed:", seed)
    random.seed(seed)

    pset = gp.PrimitiveSet("main", 8)
    pset.addPrimitive(add, 2)
    pset.addPrimitive(sub, 2)

    pset.addPrimitive(mul, 2)
    pset.addPrimitive(protectedDiv, 2)

    pset.renameArguments(ARG0="getDepth")
    pset.renameArguments(ARG1="getNConss")
    pset.renameArguments(ARG2="getNVars")
    pset.renameArguments(ARG3="getNNonz")
    pset.renameArguments(ARG4="getCutEfficacy")
    pset.renameArguments(ARG5="getCutLPSolCutoffDistance")
    pset.renameArguments(ARG6="getRowNumIntCols")
    pset.renameArguments(ARG7="getRowObjParallelism")
    pset.addTerminal(10000000)

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin,
                   pset=pset)

    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genGrow, pset=pset, min_=1, max_=17)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)

    def evaluate(scoring_function):
        tree_str = str(scoring_function)

        # Vérifier si l'arbre existe déjà dans le dictionnaire
        if tree_str in tree_scores:
            return tree_scores[tree_str],

        print(tree_str, flush=True)

        python_path = os.path.join(os.path.dirname(__file__), "subprocess_for_genetic.py")
        result = subprocess.run(
            ['python', python_path, str(scoring_function), problem, training_folder, node_select, str(time_limit),
             str(seed), str(nb_of_instances), str(fixedcutsel), str(node_lim), sol_path, str(transformed)],
            capture_output=True, text=True)
        mean_solving_time_or_gap = result.stdout

        print("mean solving time or gap: ", mean_solving_time_or_gap, flush=True)
        if mean_solving_time_or_gap == "" or mean_solving_time_or_gap == "nan":
            print("error: ", result.stderr)
            return 10e20,
        mean_solving_time_or_gap = mean_solving_time_or_gap.replace("\n", "")
        mean_solving_time_or_gap = float(mean_solving_time_or_gap)
        return mean_solving_time_or_gap,  # mean_val

    toolbox.register("evaluate", evaluate)
    toolbox.register("select", tools.selDoubleTournament, fitness_size=fitness_size, parsimony_size=parsimony_size,
                     fitness_first=True)
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
                                       halloffame=hof)  # cxpb=mate, mutpb=mutate,ngen= nb of generation

    print(pop, logbook, stats, hof)
    for elt in hof:
        print(str(elt))
    for elt in logbook:
        print(elt)
    try:
        os.makedirs(saving_folder)
    except FileExistsError:
        ""
    if saving_folder is not None:
        with open(
                saving_folder +'/'+ name + ".json",
                "w+") as outfile:
            json.dump([logbook, [str(elt) for elt in hof]], outfile)



if __name__ == "__main__":
    t_1 = time.time()



    problem = "gisp"
    problem = "miplib"
    time_limit = 10
    # problem = "wpms"
    partition = "train"
    initial_pop = 20
    nb_of_gen = 20
    fitness_size = 5
    # problem = "wpms"
    seed = random.randint(0, 10000)
    parsimony_size = 1.2
    node_select = "BFS"
    time_limit = 8
    nb_of_instances = 50
    seed = 1
    print(sys.argv)
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
            seed = sys.argv[i + 1]
        if sys.argv[i] == '-time_limit':
            time_limit = sys.argv[i + 1]
        if sys.argv[i] == '-nb_of_instances':
            nb_of_instances = sys.argv[i + 1]

        # initiate_GP(problem=problem, initial_pop=30, mate=0.9, mutate=0.1, nb_of_gen=20, seed=intialised,saving_folder="simulation_outcomes/test_for_GP_features/",training_folder="small_size")
    saving_folder = os.path.join(os.path.dirname(__file__),
                                 f'simulation_outcomes/{problem}/GP_function/')
    # training_folder = os.path.join(os.path.dirname(__file__), r'simulation_outcomes/test_for_GP_features/' )
    # name = f"{problem}_pop_{initial_pop}_mate{mate}_mutate_{mutate}_nb_gen{nb_of_gen}_seed_{seed}"

    # name = f"{problem}_pop_{initial_pop}_nb_gen{nb_of_gen}_seed_{seed}"
    if problem == "miplib":
        name = f"miplib_seed_{seed}_nb_of_instances_{nb_of_instances}_time_limit_{time_limit}"
    else:
        name = f"{problem}_pop_{initial_pop}_nb_gen{nb_of_gen}_seed_{seed}"
    main_GP(problem=problem, initial_pop=initial_pop, mate=0.9, mutate=0.1, nb_of_gen=nb_of_gen, seed=seed,
            node_select=node_select,
            saving_folder=saving_folder, name=name, training_folder=partition, fitness_size=fitness_size,
            parsimony_size=parsimony_size, time_limit=time_limit, nb_of_instances=nb_of_instances)

    print("total running time: ", time.time() - t_1)
    print(sys.argv)
