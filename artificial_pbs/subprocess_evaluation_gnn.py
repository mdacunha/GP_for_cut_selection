#!/usr/bin/env python
# coding: utf-8

# In[90]:

import json
import sys
import os
import shutil

import re
import numpy as np
import torch
from torch.multiprocessing import Process, set_start_method
from functools import partial
import argparse

from learning_to_comparenodes.utils import record_stats#, display_stats, distribute
from pathlib import Path 





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument('problem', type=str, help='problem')
    parser.add_argument('partition', type=str, help='partition')
    parser.add_argument('instance', type=str, help='instance')
    parser.add_argument('saving_folder', type=str, help='saving_folder')
    args = parser.parse_args()

    problem = args.problem
    partition = args.partition
    instance = args.instance
    saving_folder = args.saving_folder
    for nodesels in [['gnn_bfs_nprimal=2'],['gnn_bfs_nprimal=100000']]:
        n_cpu = 8
        n_instance = -1
        #nodesels = ['gnn_bfs_nprimal=2']
        normalize = True
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        #device = torch.device("cuda:0")
        verbose = False
        on_log = False
        default = False
        delete = False

        if delete:
            try:
                import shutil
                shutil.rmtree(os.path.join(os.path.abspath(''),
                                           f'stats/{problem}'))
            except:
                ''


        instances = list(Path(
            os.path.join(os.path.dirname(__file__)[:len(os.path.dirname(__file__))],
                         f"data/{problem}/{partition}/")).glob("*.lp"))
        instances = [os.path.dirname(__file__) + "/" + f"data/{problem}/{partition}/" + str(instance)]

        #Run benchmarks
        record_stats(nodesels=nodesels,instances=instances,
                                              problem=problem,
                                              device=torch.device(device),
                                              normalize=normalize,
                                              verbose=verbose,
                                              default=default)


        try:
            set_start_method('spawn')
        except RuntimeError:
            ''

        json_dir = os.path.join(os.path.abspath(''), f'stats/{problem}/{nodesels[0]}/')
        perfs_gnn = {}
        for instance_name in os.listdir(json_dir):
            with open(json_dir + instance_name, 'r') as j_file:
                perf_of_the_instance = json.load(j_file)
            perfs_gnn[instance_name[:len(instance_name)-5]] = perf_of_the_instance
        new_json_dir = os.path.join(os.path.abspath('')[:len(os.path.abspath(''))],
                                    f'{saving_folder}/{problem}/{partition}_{nodesels[0]}.json')
        with open(new_json_dir,
                  "w+") as outfile:
            json.dump(perfs_gnn, outfile)
        shutil.rmtree(json_dir)
        print("perfs are done for the function ", nodesels[0])
    print("It is ok for GNN")