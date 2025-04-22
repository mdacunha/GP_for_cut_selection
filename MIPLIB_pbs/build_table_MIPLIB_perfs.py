import os
import json
import matplotlib.pyplot as plt
import numpy as np
import conf
import math


def intersection_list(list1, list2):
   return list(set(list1) & set(list2))




def main():
    for partition in ["test","transfer"]:

        functions = ['SCIP']
        for seed in conf.seeds:
            functions.append(f"GP_seed_{seed}")
        raw_perfs = {}
        instances_that_worked = {}
        instances_that_worked_global = []
        for function in functions:
            raw_perfs[function] = {}
            for time_limit in conf.time_limits:
                extracted_infos = conf.extract_GP_info_for_MIPLIB(time_limit, function, partition)
                if extracted_infos is not None:
                    raw_perfs[function][time_limit] = extracted_infos
                    if partition == "test":
                        instances_set = conf.instances_training
                    else:
                        folder = os.path.join(conf.ROOT_DIR, f"data\\MIPLIB2017\\Instances_unziped")

                        instances_set = os.listdir(folder)
                    for instance in instances_set:
                        if instance not in raw_perfs[function][time_limit]["list_of_done"]:
                            raw_perfs[function][time_limit]["list_of_done"].append(instance)
                            raw_perfs[function][time_limit]["performances"].append([0, 1e+20])
                    if time_limit not in instances_that_worked.keys():
                        instances_that_worked[time_limit] = raw_perfs[function][time_limit]["list_of_done"]
                    else:
                        instances_that_worked[time_limit] = intersection_list(instances_that_worked[time_limit],raw_perfs[function][time_limit]["list_of_done"])
                    if instances_that_worked_global == []:
                        instances_that_worked_global = raw_perfs[function][time_limit]["list_of_done"]
                    else:
                        instances_that_worked_global =intersection_list(instances_that_worked_global,raw_perfs[function][time_limit]["list_of_done"])
                """else:
                    raw_perfs[function][time_limit] = [0, 1e+20]"""

        cleaned_perfs = {}
        for function in functions:
            cleaned_perfs[function] = {}
            for time_limit in conf.time_limits:
                if time_limit in raw_perfs[function].keys():
                    cleaned_perfs[function][time_limit] = []
                    for instance,perf in zip(raw_perfs[function][time_limit]["list_of_done"],raw_perfs[function][time_limit]["performances"]):
                        if instance in instances_that_worked_global:
                            cleaned_perfs[function][time_limit].append(perf[1])## only keep the second value, the GAP value
        markers = ['.',',','o','v','^','<','>','1','2','3','4','8','s','p','*','h','H','+','x','X','d',"D"]
        """for key in instances_that_worked.keys():
            print(len(instances_that_worked[key]))
        print(instances_that_worked_global)"""
        index_for_markers= 0
        for function in functions:
            x = []
            y = []
            for time_limit in cleaned_perfs[function].keys():
                x.append(time_limit)
                y.append(conf.shifted_geo_mean(cleaned_perfs[function][time_limit],rounding=3))
            r = np.round(np.random.rand(), 1)
            g = np.round(np.random.rand(), 1)
            b = np.round(np.random.rand(), 1)
            if function == "SCIP":
                plt.plot(x, y, label=function, marker=markers[index_for_markers], color=[r, g, b],linewidth=5)

            else:
                plt.plot(x, y, label=function, marker=markers[index_for_markers], color=[r, g, b])

            index_for_markers+=1
        plt.legend()
        plt.xlabel("time limit")
        plt.ylabel("Geometric mean GAP")
        plt.yscale('log')
        plt.title(f"performance of different GP based heuristics and SCIP, for partition {partition}")
        plt.show()


def stats_comparison_GP_SCIP(table_size="full"):
    transformed_info = {}
    for partition in ["test","transfer"]:

        functions = ["best_estimate_BFS","best_estimate_DFS","best_LB_BFS",'SCIP']
        if table_size =='full':
            seeds =conf.seeds
        else:
            seeds = conf.reduced_seeds
        for seed in seeds:
            functions.append(f"GP_seed_{seed}")
        raw_perfs = {}
        transformed_info[partition] = {}
        feasibles_for_all = {}
        for function in functions:
            raw_perfs[function] = {}
            transformed_info[partition][function] = {}
            for time_limit in conf.time_limits:
                extracted_infos = conf.extract_GP_info_for_MIPLIB(time_limit, function, partition)
                if extracted_infos is not None:
                    raw_perfs[function][time_limit] = extracted_infos
                    transformed_info[partition][function][time_limit] = {"unfeasiblies":0,"mean_geo":0,"std_geo":0}
                    list_of_feasibles=[]
                    if partition == "test":
                        instances_set = conf.instances_training
                    else:
                        folder = os.path.join(conf.ROOT_DIR, f"data\\MIPLIB2017\\Instances_unziped")

                        instances_set = os.listdir(folder)
                    for instance in instances_set:
                        perf_of_the_instance = conf.find_perf_according_to_instance(extracted_infos, instance)
                        if perf_of_the_instance is None or perf_of_the_instance[1] ==1e+20:
                            transformed_info[partition][function][time_limit]["unfeasiblies"] +=1
                        else:
                            list_of_feasibles.append(instance)

                    if time_limit not in feasibles_for_all.keys():
                        feasibles_for_all[time_limit] = list_of_feasibles
                    else:
                        feasibles_for_all[time_limit] = intersection_list(feasibles_for_all[time_limit],
                                                                              list_of_feasibles)
                else:
                    transformed_info[partition][function][time_limit] = {"unfeasiblies": 999, "mean_geo": 999, "std_geo": 999}
                    print("problem extracted info none")
        for function in functions:
            for time_limit in conf.time_limits:
                extracted_infos = conf.extract_GP_info_for_MIPLIB(time_limit, function, partition)
                if extracted_infos is not None:
                    perf_list = []
                    for instance in feasibles_for_all[time_limit]:
                        perf_of_the_instance = conf.find_perf_according_to_instance(extracted_infos, instance)
                        perf_list.append(perf_of_the_instance[1])
                    transformed_info[partition][function][time_limit]["std_geo"] = conf.geo_std(perf_list)
                    transformed_info[partition][function][time_limit]["mean_geo"] = conf.shifted_geo_mean(perf_list)

    if table_size == "full":
        considered_times_for_table =[i for i in range(10,160,20)]
    else:
        considered_times_for_table = conf.time_limits_for_paper
    mini_inf = {}
    mini_gap = {}
    for partition in ["test", "transfer"]:
        mini_inf[partition] = {}
        mini_gap[partition] = {}
        for time_limit in considered_times_for_table:
            mini_inf[partition][time_limit] = math.inf
            mini_gap[partition][time_limit] =  math.inf
            for function in functions:
                mini_inf[partition][time_limit] = min(mini_inf[partition][time_limit],transformed_info[partition][function][time_limit]["unfeasiblies"])
                mini_gap[partition][time_limit] = min(mini_gap[partition][time_limit],transformed_info[partition][function][time_limit]["mean_geo"])

    if table_size == "full":
        for partition in ["test", "transfer"]:
            text = ""
            for time_limit in considered_times_for_table:
                text += f"&\multicolumn{{2}}{{c}}{{{time_limit} seconds}}"
            text += "\\\\"
            print(text)
            text = ""
            for time_limit in considered_times_for_table:
                text += f"&\\textsc{{Inf}} & \\textsc{{Gap}}"
            text += "\\\\"
            print(text)
            for function in functions:

                text = conf.proper_names[function]
                for time_limit in considered_times_for_table:
                    c_inf, m_inf, y_inf, k_inf = conf.gradient_color(
                        transformed_info[partition][function][time_limit]["unfeasiblies"],
                        mini_inf[partition][time_limit],
                        mini_inf[partition][time_limit] * 1.4)
                    c_gap, m_gap, y_gap, k_gap = conf.gradient_color(
                        transformed_info[partition][function][time_limit]["mean_geo"], mini_gap[partition][time_limit],
                        mini_gap[partition][time_limit] * 1.4)
                    text += f' & \cellcolor[cmyk]{{{c_inf},{m_inf},{y_inf},{k_inf}}}$ {str(transformed_info[partition][function][time_limit]["unfeasiblies"])}$ & \cellcolor[cmyk]{{{c_gap},{m_gap},{y_gap},{k_gap}}}${str(transformed_info[partition][function][time_limit]["mean_geo"])}$$\pm$${str(transformed_info[partition][function][time_limit]["std_geo"])}$'

                text += '\\\\'
                print(text)
            print("########################")
    else:
        for function in functions:

            text = conf.proper_names[function]
            for partition in ["test","transfer"]:

                for time_limit in considered_times_for_table:
                    c_inf,m_inf,y_inf,k_inf = conf.gradient_color(transformed_info[partition][function][time_limit]["unfeasiblies"], mini_inf[partition][time_limit],
                                                  mini_inf[partition][time_limit]* 1.4)
                    c_gap, m_gap, y_gap, k_gap = conf.gradient_color(
                        transformed_info[partition][function][time_limit]["mean_geo"], mini_gap[partition][time_limit],
                        mini_gap[partition][time_limit] * 1.4)
                    text += f' & \\textcolor[cmyk]{{{c_inf},{m_inf},{y_inf},{k_inf}}}{{$ {str(transformed_info[partition][function][time_limit]["unfeasiblies"])}$}} & \\textcolor[cmyk]{{{c_gap},{m_gap},{y_gap},{k_gap}}}${{{str(transformed_info[partition][function][time_limit]["mean_geo"])}\pm{str(transformed_info[partition][function][time_limit]["std_geo"])}$}}'

            text += '\\\\'
            print(text)

if __name__ == "__main__":
    stats_comparison_GP_SCIP("full")#full or reduced table
