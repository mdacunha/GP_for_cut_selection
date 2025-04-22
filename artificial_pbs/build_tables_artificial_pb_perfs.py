import matplotlib.pyplot as plt
import re
import os
import json
import conf

from conf import *

def plot_one_index_one_feature(perfs,index):
    x = []
    y = []
    for key in perfs:
        x.append(str(key[:len(key) - 37]))  # instance name
        y.append(perfs[key][index])  # nb of nodes
    return x,y
def gather_info_from_json_files(problems=["fcmcnf","wpms","gisp"],partitions=["test","transfer"],saving_folder="simulation_outcomes"):
    dic_info = {}
    for problem in problems:#"wpms","gisp",
        dir = os.path.join(conf.ROOT_DIR,
                           f'{saving_folder}/{problem}/')
        dic_info[problem] = {}
        for partition in partitions:
            dic_info[problem][partition] = {}
            for one_perf_file in os.listdir(dir):
                if re.match(partition, one_perf_file):
                    one_perf_path = dir+ str(one_perf_file)
                    with open(
                            one_perf_path,
                            'r') as openfile:
                        perfs = json.load(openfile)
                    x,y = plot_one_index_one_feature(perfs,1)
                    function = one_perf_file[len(partition) + 1: len(one_perf_file) - 5]
                    #plt.plot(x,y,label=function)
                    print(f"geometric mean solving time for set {partition} for method {function} is ${shifted_geo_mean(y, 1)}\\pm {geo_std(y)}$ on {len(perfs)} instances")

                    dic_info[problem][partition][function] = [shifted_geo_mean(y, 1, rounding=1),geo_std(y, rounding=1)]


    return dic_info


def just_get_the_output_results(dic_info):

    problems = list(dic_info.keys())
    partitions = list(dic_info[problems[0]].keys())
    functions = list(dic_info[problems[0]][partitions[0]].keys())
    for problem in problems:
        for partition in partitions:
            string_to_print = f"RESULTS FOR {problem} - {partition}\n"
            for function in functions:
                string_to_print+= f"results for function {proper_names[function]} are {dic_info[problem][partition][function][0]} +-{dic_info[problem][partition][function][1]}\n"
            print(string_to_print)
def build_table_with_cell_colors(dic_info):
    min_elt = {}
    for problem in ["fcmcnf", "wpms", "gisp"]:
        min_elt[problem] = {}
        for partition in ["test", "transfer"]:
            min_elt[problem][partition] = 10000000000
            for function in dic_info[problem][partition].keys():
                min_elt[problem][partition] = min(min_elt[problem][partition],dic_info[problem][partition][function][0])
    for function in dic_info["fcmcnf"]["test"].keys():

        string_to_print = proper_names[function]
        for problem in ["fcmcnf", "wpms", "gisp"]:
            for partition in ["test", "transfer"]:
                c,m,y,k = conf.gradient_color(dic_info[problem][partition][function][0], min_elt[problem][partition], min_elt[problem][partition] *1.4)
                string_to_print += f"& \\textcolor[cmyk]{{{c},{m},{y},{k}}}{{${dic_info[problem][partition][function][0]}\\pm{dic_info[problem][partition][function][1]}$}}$"
        string_to_print += "\\\\"
        print(string_to_print)
if __name__ == "__main__":
    dic_info = gather_info_from_json_files()

    build_table_with_cell_colors(dic_info)