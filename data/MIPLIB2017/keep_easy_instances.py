import os

import conf
from scip_solver import perform_SCIP_instance
if __name__ == "__main__":
    time_limit = 10
    parameter_settings = True
    node_comp = "SCIP"
    instances_folder = os.path.join(conf.ROOT_DIR, f"data/MIPLIB2017/Instances_unziped")
    reduced_instance_set = []
    for instance in os.listdir(instances_folder):
            if instance.endswith(".lp") or instance.endswith(".mps"):
                # print(instance)
                # print(instance)
                instance_path = os.path.join(os.path.dirname(__file__), instances_folder +"/"+ str(instance))
                # instance_path = os.path.dirname(__file__) + "/" + instances_folder + str(instance)
                # print(instance_path)
                nb_visited_nodes, time = perform_SCIP_instance(instance_path, node_comp, node_select="",
                                                               parameter_settings=parameter_settings,time_limit=time_limit)
                if nb_visited_nodes>1:
                    reduced_instance_set.append(instance)
    print(reduced_instance_set)