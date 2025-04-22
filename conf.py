import os
import numpy as np
import scipy
import json
import math


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))




def shifted_geo_mean(iterable,shift=1,rounding=2):
    if rounding == 0:
        return round(np.exp(np.mean(np.log(np.array(iterable) + shift))) - shift)
    return round(np.exp(np.mean(np.log(np.array(iterable) + shift ))) - shift,rounding)
def geo_std(a,rounding=1):
    if rounding == 0:
        return round(np.exp(np.sqrt(np.mean((np.log(np.array(a) + 1) - np.log(shifted_geo_mean(a, 1))) ** 2))))
    return round(np.exp(np.sqrt(np.mean(  ( np.log(np.array(a)+1) - np.log(shifted_geo_mean(a,1)) )**2 ))),rounding)


def protectedDiv(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1

def gradient_color(value, min_val, max_val):### used for the plots
    if math.isnan(value):
        return (1,1,1)
    # Ensure the value is within the range [min_val, max_val]
    value = max(min_val, min(value, max_val))

    normalized_value = (value - min_val) / (max_val - min_val)

    c = round( 1 * (1 - normalized_value) ,2)
    m =round(normalized_value *1,2)
    y = 1
    k = round(0.25*normalized_value,2)

    return (c,m,y,k)








########## SPECIFIC TO ARTICIFIAL PROBLEMS

gp_funcs_artificial_problems = {'gisp': {'1.4': "sub(protectedDiv(getDepth, getEstimate), getDepth)",
                         "1.2": "sub(protectedDiv(getDepth, getEstimate), getDepth)"},
                "wpms": {'1.4': "protectedDiv(protectedDiv(protectedDiv(getDepth, getLowerbound), getNVars), getDepth)",
                         "1.2": "protectedDiv(getNConss, add(getEstimate, 10000000))"},
                "fcmcnf": {'1.4': "add(getDepth, protectedDiv(getEstimate, getDepth))",
                           "1.2": "protectedDiv(getEstimate, getDepth)"}}




########## SPECIFIC TO MIPLIB ##############
proper_names = {'best_estimate_BFS': "BE BFS","best_estimate_DFS": "BE DFS","best_LB_BFS": "LB BFS","gnn_bfs_nprimal=100000": "GNN full", "gnn_bfs_nprimal=2": "GNN 2 dives" , "GP_parsimony_parameter_1.2": "GP2S","GP_parsimony_parameter_1.4":"to remove","SCIP":"SCIP"}

for seed in range(1,21):
    proper_names[f"GP_seed_{seed}"] = f"GP2S {seed}"


def extract_GP_info_for_MIPLIB(time_limit, function, partition):
    folder = os.path.join(os.path.dirname(__file__), f"simulation_outcomes\\MIPLIB")
    path =folder + f'\\time_limit_{time_limit}_function_{function}_partition_{partition}.json'
    isExist = os.path.exists(path)
    if isExist:
        with open(path
                ,
                'r') as openfile:
            perfs = json.load(openfile)
        return perfs
    else:
        return None

def find_perf_according_to_instance(info,instance):
    for instance_of_folder,perf in zip(info["list_of_done"],info["performances"]):
        if instance_of_folder == instance:
            return perf
    return None

gp_funcs_MIPLIB_seeds = {'1':'add(getLowerbound, protectedDiv(getNVars, sub(getDepth, getNVars)))',
                '2':'add(add(getEstimate, getLowerbound), getNConss)',
                '3':'add(getLowerbound, protectedDiv(getDualboundRoot, getEstimate))',
                '4':'add(protectedDiv(getLowerbound, getDualboundRoot), mul(getNVars, getDepth))',
                '5':'protectedDiv(add(add(getEstimate, getNVars), getNVars), sub(getNConss, sub(getNVars, protectedDiv(getLowerbound, getNVars))))',
                '6':'mul(add(getEstimate, getLowerbound), add(getEstimate, getLowerbound))',
                '7':'add(add(mul(getLowerbound, getEstimate), getNVars), getEstimate)',
                '8':'sub(mul(getDualboundRoot, sub(add(10000000, sub(10000000, getLowerbound)), getEstimate)), mul(getDepth, getLowerbound))',
                '9':'protectedDiv(getDualboundRoot, protectedDiv(10000000, getEstimate))',
                '10':'protectedDiv(getLowerbound, protectedDiv(getNConss, getDepth))',
                '11': 'sub(protectedDiv(getDualboundRoot, getLowerbound), getNConss)',
                '12': 'protectedDiv(getLowerbound, mul(getDepth, add(add(getNVars, getEstimate), getDepth)))',
                '13': 'add(mul(sub(mul(getLowerbound, getLowerbound), getNVars), getNConss), protectedDiv(mul(getNVars, 10000000), mul(sub(mul(10000000, sub(getDepth, getDepth)), add(10000000, mul(getEstimate, sub(10000000, getNVars)))), add(getLowerbound, 10000000))))',
                '14': 'mul(mul(getLowerbound, protectedDiv(protectedDiv(getDepth, getLowerbound), getLowerbound)), sub(getEstimate, getNVars))',
                '15': 'mul(add(mul(add(add(getEstimate, add(getNVars, getEstimate)), getEstimate), getNVars), getEstimate), getEstimate)',
                '16': 'add(getLowerbound, getLowerbound)',
                '17': 'add(getEstimate, getLowerbound)',
                '18': 'add(sub(10000000, mul(getLowerbound, getDepth)), getEstimate)',
                '19': 'add(mul(getLowerbound, getDepth), protectedDiv(getEstimate, protectedDiv(protectedDiv(10000000, getDualboundRoot), protectedDiv(mul(getNVars, getDualboundRoot), 10000000))))',
                '20':'protectedDiv(protectedDiv(sub(getNVars, getNConss), 10000000), getEstimate)'
                }

nbs_of_instances = [200,100,50]
time_limits = [10,30,50,70,90,110,130,150]
time_limits_for_paper = [10,50,150]#100,
seeds = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
reduced_seeds = [2,4,6,8,10,12,14,16,18,20]



instances_training = ['10teams.mps', '22433.mps', '23588.mps', 'acc-tight5.mps', 'app1-1.mps', 'app3.mps', 'assign1-5-8.mps', 'b-ball.mps', 'bab1.mps', 'beasleyC1.mps', 'berlin_5_8_0.mps', 'bg512142.mps', 'bienst1.mps', 'bienst2.mps', 'binkar10_1.mps', 'blend2.mps', 'bnatt400.mps', 'bppc4-08.mps', 'bppc8-09.mps', 'ci-s4.mps', 'cost266-UUE.mps', 'csched007.mps', 'csched008.mps', 'csched010.mps', 'danoint.mps', 'dcmulti.mps', 'drayage-25-27.mps', 'dsbmip.mps', 'fastxgemm-n2r6s0t2.mps', 'fastxgemm-n2r7s4t1.mps', 'g200x740.mps', 'gen-ip002.mps', 'gen-ip016.mps', 'gen-ip021.mps', 'gen-ip036.mps', 'gen-ip054.mps', 'ger50-17-ptp-pop-6t.mps', 'germany50-UUM.mps', 'glass4.mps', 'gmu-35-40.mps', 'gmu-35-50.mps', 'graphdraw-domain.mps', 'graphdraw-gemcutter.mps', 'graphdraw-mainerd.mps', 'gsvm2rl5.mps', 'gus-sch.mps', 'ic97_potential.mps', 'ic97_tension.mps', 'icir97_potential.mps', 'icir97_tension.mps', 'k16x240b.mps', 'khb05250.mps', 'l2p12.mps', 'lectsched-4-obj.mps', 'leo1.mps', 'leo2.mps', 'loopha13.mps', 'mad.mps', 'markshare1.mps', 'markshare2.mps', 'markshare_4_0.mps', 'markshare_5_0.mps', 'mas74.mps', 'mas76.mps', 'mcsched.mps', 'mik-250-20-75-3.mps', 'mik-250-20-75-5.mps', 'milo-v12-6-r2-40-1.mps', 'milo-v13-4-3d-3-0.mps', 'milo-v13-4-3d-4-0.mps', 'mkc.mps', 'n5-3.mps', 'n6-3.mps', 'n7-3.mps', 'neos-1067731.mps', 'neos-1423785.mps', 'neos-1442119.mps', 'neos-1582420.mps', 'neos-2624317-amur.mps', 'neos-2652786-brda.mps', 'neos-2657525-crna.mps', 'neos-3009394-lami.mps', 'neos-3024952-loue.mps', 'neos-3046601-motu.mps', 'neos-3046615-murg.mps', 'neos-3072252-nete.mps', 'neos-3083819-nubu.mps', 'neos-3118745-obra.mps', 'neos-3426085-ticino.mps', 'neos-3426132-dieze.mps', 'neos-3530903-gauja.mps', 'neos-3530905-gaula.mps', 'neos-3660371-kurow.mps', 'neos-3691541-lonja.mps', 'neos-3754480-nidda.mps', 'neos-4333596-skien.mps', 'neos-4393408-tinui.mps', 'neos-4650160-yukon.mps', 'neos-4738912-atrato.mps', 'neos-480878.mps', 'neos-4954672-berkel.mps', 'neos-5045105-creuse.mps', 'neos-5051588-culgoa.mps', 'neos-5107597-kakapo.mps', 'neos-5140963-mincio.mps', 'neos-5178119-nalagi.mps', 'neos-5182409-nasivi.mps', 'neos-5261882-treska.mps', 'neos-555424.mps', 'neos-555884.mps', 'neos-631517.mps', 'neos-686190.mps', 'neos-831188.mps', 'neos-911970.mps', 'neos-933562.mps', 'neos16.mps', 'neos5.mps', 'newdano.mps', 'nh97_potential.mps', 'nh97_tension.mps', 'ns1830653.mps', 'ns2071214.mps', 'nsa.mps', 'nu120-pr12.mps', 'opm2-z6-s1.mps', 'p500x2988.mps', 'pigeon-10.mps', 'pigeon-13.mps', 'pigeon-16.mps', 'pigeon-20.mps', 'pk1.mps', 'probportfolio.mps', 'prod1.mps', 'prod2.mps', 'qiu.mps', 'r50x360.mps', 'ran12x21.mps', 'ran13x13.mps', 'ran14x18-disj-8.mps', 'rentacar.mps', 'rocI-4-11.mps', 'rococoC10-001000.mps', 'roll3000.mps', 'sct2.mps', 'set3-09.mps', 'set3-10.mps', 'set3-16.mps', 'set3-20.mps', 'sp98ir.mps', 'supportcase20.mps', 'supportcase39.mps', 'swath.mps', 'swath2.mps', 'swath3.mps', 'tanglegram6.mps', 'timtab1.mps', 'timtab1CUTS.mps', 'tr12-30.mps', 'traininstance2.mps', 'traininstance6.mps', 'tw-myciel4.mps', 'uct-subprob.mps', 'umts.mps', 'usAbbrv-8-25_70.mps', 'v150d30-2hopcds.mps', 'wachplan.mps']

if __name__ == "__main__":
    ### modify the GP_function build from MIPLIB to the symbols used in the paper.
    for seed in gp_funcs_MIPLIB_seeds.keys():
        function = gp_funcs_MIPLIB_seeds[seed]
        function = function.replace("getDepth","d_i")
        function = function.replace("getEstimate", "BE_i")
        function = function.replace("getLowerbound", "z_i")
        function = function.replace("getDualboundRoot", "z_0")
        function = function.replace("getNConss", "m")
        function = function.replace("getNVars", "n")
        print(f"${seed}$ & ${function}$\\\\")