from pathlib import Path
import os

import pyscipopt
from pyscipopt import Model, quicksum, Nodesel, SCIP_PRESOLTIMING, SCIP_PROPTIMING
import statistics
from operator import *
import argparse
import numpy as np
import time

from conf import *
from cut_selection_policies import CustomCutSelector
from ConstraintHandler_GP import RepeatSepaConshdlr


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
        # Create a dummy constraint handler that forces the num_rounds amount of separation rounds
        num_rounds = 50
        constraint_handler = RepeatSepaConshdlr(model, num_rounds)
        model.includeConshdlr(constraint_handler, "RepeatSepa", "Forces a certain number of separation rounds",
                             sepapriority=-1, enfopriority=1, chckpriority=-1, sepafreq=-1, propfreq=-1,
                             eagerfreq=-1, maxprerounds=-1, delaysepa=False, delayprop=False, needscons=False,
                             presoltiming=SCIP_PRESOLTIMING.FAST, proptiming=SCIP_PROPTIMING.AFTERLPNODE)
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

def set_scip_separator_params(scip, max_rounds_root=-1, max_rounds=-1, max_cuts_root=10000, max_cuts=10000,
                              frequency=10):
    """
    Function for setting the separator params in SCIP. It goes through all separators, enables them at all points
    in the solving process,
    Args:
        scip: The SCIP Model object
        max_rounds_root: The max number of separation rounds that can be performed at the root node
        max_rounds: The max number of separation rounds that can be performed at any non-root node
        max_cuts_root: The max number of cuts that can be added per round in the root node
        max_cuts: The max number of cuts that can be added per node at any non-root node
        frequency: The separators will be called each time the tree hits a new multiple of this depth
    Returns:
        The SCIP Model with all the appropriate parameters now set
    """

    assert type(max_cuts) == int and type(max_rounds) == int
    assert type(max_cuts_root) == int and type(max_rounds_root) == int

    # First for the aggregation heuristic separator
    scip.setParam('separating/aggregation/freq', frequency)
    scip.setParam('separating/aggregation/maxrounds', max_rounds)
    scip.setParam('separating/aggregation/maxroundsroot', max_rounds_root)
    scip.setParam('separating/aggregation/maxsepacuts', 1000)
    scip.setParam('separating/aggregation/maxsepacutsroot', 1000)

    # Now the Chvatal-Gomory w/ MIP separator
    # scip.setParam('separating/cgmip/freq', frequency)
    # scip.setParam('separating/cgmip/maxrounds', max_rounds)
    # scip.setParam('separating/cgmip/maxroundsroot', max_rounds_root)

    # The clique separator
    scip.setParam('separating/clique/freq', frequency)
    scip.setParam('separating/clique/maxsepacuts', 1000)

    # The close-cuts separator
    scip.setParam('separating/closecuts/freq', frequency)

    # The CMIR separator
    scip.setParam('separating/cmir/freq', frequency)

    # The Convex Projection separator
    scip.setParam('separating/convexproj/freq', frequency)
    scip.setParam('separating/convexproj/maxdepth', -1)

    # The disjunctive cut separator
    scip.setParam('separating/disjunctive/freq', frequency)
    scip.setParam('separating/disjunctive/maxrounds', max_rounds)
    scip.setParam('separating/disjunctive/maxroundsroot', max_rounds_root)
    scip.setParam('separating/disjunctive/maxinvcuts', 1000)
    scip.setParam('separating/disjunctive/maxinvcutsroot', 1000)
    scip.setParam('separating/disjunctive/maxdepth', -1)

    # The separator for edge-concave function
    scip.setParam('separating/eccuts/freq', frequency)
    scip.setParam('separating/eccuts/maxrounds', max_rounds)
    scip.setParam('separating/eccuts/maxroundsroot', max_rounds_root)
    scip.setParam('separating/eccuts/maxsepacuts', 1000)
    scip.setParam('separating/eccuts/maxsepacutsroot', 1000)
    scip.setParam('separating/eccuts/maxdepth', -1)

    # The flow cover cut separator
    scip.setParam('separating/flowcover/freq', frequency)

    # The gauge separator
    scip.setParam('separating/gauge/freq', frequency)

    # Gomory MIR cuts
    scip.setParam('separating/gomory/freq', frequency)
    scip.setParam('separating/gomory/maxrounds', max_rounds)
    scip.setParam('separating/gomory/maxroundsroot', max_rounds_root)
    scip.setParam('separating/gomory/maxsepacuts', 1000)
    scip.setParam('separating/gomory/maxsepacutsroot', 1000)

    # The implied bounds separator
    scip.setParam('separating/impliedbounds/freq', frequency)

    # The integer objective value separator
    scip.setParam('separating/intobj/freq', frequency)

    # The knapsack cover separator
    scip.setParam('separating/knapsackcover/freq', frequency)

    # The multi-commodity-flow network cut separator
    scip.setParam('separating/mcf/freq', frequency)
    scip.setParam('separating/mcf/maxsepacuts', 1000)
    scip.setParam('separating/mcf/maxsepacutsroot', 1000)

    # The odd cycle separator
    scip.setParam('separating/oddcycle/freq', frequency)
    scip.setParam('separating/oddcycle/maxrounds', max_rounds)
    scip.setParam('separating/oddcycle/maxroundsroot', max_rounds_root)
    scip.setParam('separating/oddcycle/maxsepacuts', 1000)
    scip.setParam('separating/oddcycle/maxsepacutsroot', 1000)

    # The rapid learning separator
    scip.setParam('separating/rapidlearning/freq', frequency)

    # The strong CG separator
    scip.setParam('separating/strongcg/freq', frequency)

    # The zero-half separator
    scip.setParam('separating/zerohalf/freq', frequency)
    scip.setParam('separating/zerohalf/maxcutcands', 100000)
    scip.setParam('separating/zerohalf/maxrounds', max_rounds)
    scip.setParam('separating/zerohalf/maxroundsroot', max_rounds_root)
    scip.setParam('separating/zerohalf/maxsepacuts', 1000)
    scip.setParam('separating/zerohalf/maxsepacutsroot', 1000)

    # The rlt separator
    scip.setParam('separating/rlt/freq', frequency)
    scip.setParam('separating/rlt/maxncuts', 1000)
    scip.setParam('separating/rlt/maxrounds', max_rounds)
    scip.setParam('separating/rlt/maxroundsroot', max_rounds_root)

    # Now the general cut and round parameters
    scip.setParam("separating/maxroundsroot", max_rounds_root - 1)
    scip.setParam("separating/maxstallroundsroot", max_rounds_root - 1)
    scip.setParam("separating/maxcutsroot", 10000)

    scip.setParam("separating/maxrounds", 0)
    scip.setParam("separating/maxstallrounds", 0)
    scip.setParam("separating/maxcuts", 0)

    return scip