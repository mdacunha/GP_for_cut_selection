import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "GNN_method"))

from pyscipopt import Model, SCIP_PRESOLTIMING, SCIP_PROPTIMING, SCIP_PARAMSETTING
from operator import *
import numpy as np
import time

from GNN_method.utilities import get_filename
from conf import *
from cut_selection_policies import CustomCutSelector
from constraintHandler_GP import RepeatSepaConshdlr

#from GNN_method.Slurm.train_neural_network import get_standard_solve_data

from RL.arguments import args
import torch
from RL.neural_network import nnet as nnet_for_GP

def perform_SCIP_instance(instance_path, cut_comp="estimate", node_select="BFS", parameter_settings=False,
                          time_limit=0, fixedcutsel=False, node_lim=-1, sol_path=None, is_Test=False, final_test=False,
                          test=False, num_cuts_per_round=10, RL=False, higher_simulation_folder="", 
                          nnet=None, inputs_type="", heuristic=False, get_scores=False):
    
    if RL and nnet==None:
        new_args = args
        new_args.update({
                'inputs_type': inputs_type
            })
        nnet = nnet_for_GP(new_args)
        filepath = os.path.join(higher_simulation_folder, "weights" + ".pth.tar")
        if not os.path.exists(filepath):
            raise ("No model in path {}".format(filepath))
        map_location = None if args["cuda"] else 'cpu'
        checkpoint = torch.load(filepath, map_location=map_location)
        nnet.load_state_dict(checkpoint['state_dict'])
        if args["cuda"]:
            nnet.cuda()
        
    model = Model()
    model.hideOutput()
    if fixedcutsel:            
        model.setParam('limits/nodes', node_lim)
        model.setParam('presolving/maxrounds', 1)
        model.setParam('estimation/restarts/restartlimit', 0)
        model.setParam('estimation/restarts/restartpolicy', 'n')
        model.setParam('presolving/maxrestarts', 0)
        model.disablePropagation()
    if parameter_settings:
        model.setParam('constraints/linear/upgrade/logicor', 0)
        model.setParam('constraints/linear/upgrade/indicator', 0)
        model.setParam('constraints/linear/upgrade/knapsack', 0)
        model.setParam('constraints/linear/upgrade/setppc', 0)
        model.setParam('constraints/linear/upgrade/xor', 0)
        model.setParam('constraints/linear/upgrade/varbound', 0)
    if cut_comp == "SCIP":
        pass
    else:
        # Create a dummy constraint handler that forces the num_rounds amount of separation rounds
        num_rounds = 50
        if fixedcutsel:
            constraint_handler = RepeatSepaConshdlr(model, num_rounds)
            model.includeConshdlr(constraint_handler, "RepeatSepa", "Forces a certain number of separation rounds",
                                    sepapriority=-1, enfopriority=1, chckpriority=-1, sepafreq=-1, propfreq=-1,
                                    eagerfreq=-1, maxprerounds=-1, delaysepa=False, delayprop=False, needscons=False,
                                    presoltiming=SCIP_PRESOLTIMING.FAST, proptiming=SCIP_PROPTIMING.AFTERLPNODE)
            cut_selector = CustomCutSelector(comp_policy=cut_comp, num_cuts_per_round=num_cuts_per_round, test=test, RL=RL,
                                             get_scores=get_scores, heuristic=heuristic)
            model.includeCutsel(cut_selector, "", "", 536870911)
            model.setParam('separating/maxstallroundsroot', num_rounds)
            model = set_scip_separator_params(model, num_rounds, 0, num_cuts_per_round, 0, 0)
        else:
            #cut_selector = CustomCutSelector(comp_policy=cut_comp)
            cut_selector = CustomCutSelector(comp_policy=cut_comp, num_cuts_per_round=num_cuts_per_round, test=test, RL=RL, 
                                             nnet=nnet, inputs_type=inputs_type, is_Test=is_Test, final_test=final_test,
                                             get_scores=get_scores, heuristic=heuristic)
            model.includeCutsel(cut_selector, "", "", 536870911)
            

    if fixedcutsel:
        model.setHeuristics(SCIP_PARAMSETTING.OFF)
        model.setParam('branching/leastinf/priority', 10000000)

    if time_limit != 0:
        model.setRealParam("limits/time", time_limit)
    
    model.readProblem(instance_path)
    
    if sol_path != "None" and sol_path is not None:
        real_sol_path = get_filename(sol_path, instance_path.split("/")[-1].split("__trans")[0], 1, trans=True, root=False, sample_i=None, ext='sol')
        assert os.path.isfile(real_sol_path) and '.sol' in real_sol_path, 'Sol is {}'.format(real_sol_path)
        sol = model.readSolFile(real_sol_path)
        model.addSol(sol)
    model.optimize()

    """if not os.path.exists(f"gp_stats_{num_cuts_per_round}.txt") and cut_comp != "SCIP" and is_Test:
        stats_file = f"gp_stats_{num_cuts_per_round}.txt"
        model.writeStatistics(stats_file)
    if not os.path.exists(f"scip_stats_{num_cuts_per_round}.txt") and cut_comp == "SCIP" and is_Test:
        stats_file = f"scip_stats_{num_cuts_per_round}.txt"
        model.writeStatistics(stats_file)"""

    if time_limit != 0:
        if fixedcutsel and is_Test:
            score = 0
            root_path = os.path.join(ROOT_DIR, "GNN_method", "RootResults/")
            standard_solve_data = get_standard_solve_data(root_path, root=True)
            sd = standard_solve_data[instance_path.split("/")[-1].split("__")[0]][1]
            bd = model.getGap()
            score = (sd["gap"] - bd) / (abs(sd["gap"]) + 1e-8)
            return model.getNNodes(), score
        return model.getNNodes(), model.getGap(), cut_selector.k_list(), None
    if cut_comp == "SCIP":
        return model.getNNodes(), model.getSolvingTime(), None, None
    try:
        t = cut_selector.time()/model.getSolvingTime()*100
    except ZeroDivisionError:
        t = 0
    return model.getNNodes(), model.getSolvingTime(), cut_selector.k_list(), t


def perform_SCIP_instances_using_a_tuned_comp_policy(instances_folder="", cut_comp="estimate", node_select="BFS",
                                                     nb_max=None,
                                                     printing=False,
                                                     parameter_settings=False, 
                                                     fixedcutsel=False,
                                                     node_lim=-1,
                                                     time_limit=None,
                                                     instances_indexes=None,
                                                     sol_path=None,
                                                     test=False,
                                                     num_cuts_per_round=10,
                                                     RL=False,
                                                     inputs_type="",
                                                     higher_simulation_folder="",
                                                     heuristic=False):  
    sol_times = []
    nnodes = []
    nb_done = 0
    index_of_the_folder = 0
    for instance in os.listdir(instances_folder):
        if instances_indexes is None or index_of_the_folder in instances_indexes:
            if instance.endswith(".lp") or instance.endswith(".mps"):

                instance_path = os.path.join(os.path.dirname(__file__), instances_folder + "/" + str(instance))

                visited_nodes, time_or_gap, _, t = perform_SCIP_instance(instance_path, cut_comp, node_select,
                                                                   parameter_settings=parameter_settings,
                                                                   time_limit=time_limit, fixedcutsel=fixedcutsel, 
                                                                   node_lim=node_lim, sol_path=sol_path, test=test,
                                                                   num_cuts_per_round=num_cuts_per_round, RL=RL,
                                                                   inputs_type=inputs_type,
                                                                   higher_simulation_folder=higher_simulation_folder,
                                                                   heuristic=heuristic)
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