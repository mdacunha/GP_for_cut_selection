from pathlib import Path
import os

import pyscipopt
from pyscipopt import Model, quicksum, Nodesel
import statistics
from operator import *
import argparse
import numpy as np
import time

from conf import *
from cut_selection_policies import CustomCutSelector


def perform_SCIP_instance(instance_path, cut_comp="estimate", node_select="BFS", parameter_settings=False,
                          time_limit=None):
    model = Model()
    model.hideOutput()
    model.readProblem(instance_path)
    optsol = None
    if parameter_settings:
        model.setParam('constraints/linear/upgrade/logicor', 0)
        model.setParam('constraints/linear/upgrade/indicator', 0)
        model.setParam('constraints/linear/upgrade/knapsack', 0)
        model.setParam('constraints/linear/upgrade/setppc', 0)
        model.setParam('constraints/linear/upgrade/xor', 0)
        model.setParam('constraints/linear/upgrade/varbound', 0)
    if time_limit is not None:
        model.setRealParam("limits/time", time_limit)
    if cut_comp == "SCIP":
        pass
    else:
        cut_selector = CustomCutSelector(comp_policy=cut_comp)
        model.includeCutsel(cut_selector, "", "", 536870911)

    model.optimize()
    if time_limit is not None:
        return model.getNNodes(), model.getGap()
    return model.getNNodes(), model.getSolvingTime()


def perform_SCIP_instances_using_a_tuned_comp_policy(instances_folder="", cut_comp="estimate", node_select="BFS",
                                                     nb_max=None,
                                                     printing=False,
                                                     parameter_settings=False, time_limit=None,
                                                     instances_indexes=None):  # comp policy is either a str LB or estimate of a function
    sol_times = []
    nnodes = []
    nb_done = 0
    index_of_the_folder = 0
    for instance in os.listdir(instances_folder):
        if instances_indexes is None or index_of_the_folder in instances_indexes:
            if instance.endswith(".lp") or instance.endswith(".mps"):

                instance_path = os.path.join(os.path.dirname(__file__), instances_folder + "/" + str(instance))

                visited_nodes, time_or_gap = perform_SCIP_instance(instance_path, cut_comp, node_select,
                                                                   parameter_settings=parameter_settings,
                                                                   time_limit=time_limit)
                sol_times.append(time_or_gap)
                nnodes.append(visited_nodes)
                nb_done += 1
                if index_of_the_folder == 261:
                    time.sleep(1000000)

                if nb_max is not None and nb_done >= nb_max:
                    mean_val = shifted_geo_mean(sol_times)
                    meannnodes = shifted_geo_mean(nnodes)
                    return (meannnodes, mean_val)
        index_of_the_folder += 1
    mean_val = shifted_geo_mean(sol_times)
    meannnodes = shifted_geo_mean(nnodes)

    return (meannnodes, mean_val)

