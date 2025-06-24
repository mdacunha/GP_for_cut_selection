import os
from tqdm import tqdm

import numpy as np
from conf import *
import matplotlib.pyplot as plt

import torch
import torch.optim as optim

from RL.neural_network import nnet
from scip_solver import perform_SCIP_instance
from RL.arguments import args

import multiprocessing
from functools import partial

class NeuralNetworkWrapper():
    def __init__(self, training_path="", testing_path="", higher_simulation_folder="", problem="gisp", cut_comp="", 
                 parameter_settings=False, saving_folder="", load_checkpoint=False, inputs_type="", sol_path=None):
        
        new_args = args
        self.inputs_type = inputs_type
        new_args.update({
                'inputs_type': self.inputs_type
            })
        self.nnet = nnet(new_args)
        self.training_path = training_path
        self.testing_path = testing_path
        self.higher_simulation_folder = higher_simulation_folder
        self.problem = problem
        self.cut_comp = cut_comp
        self.parameter_settings = parameter_settings
        self.saving_folder = saving_folder
        self.sol_path = sol_path
        if load_checkpoint:
            self.load_checkpoint(folder=self.higher_simulation_folder, filename=self.saving_folder)
        if args["cuda"]:
            self.nnet.cuda()     

    def learn(self):
        best_test_score = np.inf
        for epoch in range(args["epochs"]):
            print('EPOCH ::: ' + str(epoch + 1), flush=True)
            self.train()
            test_score = self.test()
            print("Loss for test instances :", test_score, flush=True)
            if test_score < best_test_score:
                best_test_score = test_score
                print("saving model...", flush=True)
                self.save_checkpoint(folder=self.higher_simulation_folder, filename=self.saving_folder)

    def train(self):

        optimizer = optim.Adam(self.nnet.parameters(), lr=args['lr'])

        instances = [os.path.join(self.training_path[0], f) for f in os.listdir(self.training_path[0])]
        if len(self.training_path)>1:
            instances += [os.path.join(self.training_path[1], f) for f in os.listdir(self.training_path[1])]

        batch_count = int(len(instances) / args['batch_size']) + (len(instances) % args['batch_size'] != 0)
        t = tqdm(range(batch_count), desc="Training Net")
        for i in t:
            optimizer.zero_grad()

            instance_args = [(instance, self.cut_comp, self.parameter_settings, self.sol_path, self.inputs_type, self.nnet, "train") 
                            for instance in instances[i * args['batch_size'] : (i+1) * args['batch_size']]]
            with multiprocessing.Pool(processes=min(multiprocessing.cpu_count(), len(instances))) as pool:
                results = pool.map(self.process_instance, instance_args)
            
            examples = [(r,k) for (r,k) in results if r is not None]
            if not examples:
                print("Aucun exemple valide trouvé.")
                return
        

            score = shifted_geo_mean(examples)
        
            t.set_postfix(Loss=float(score))
            
            score = torch.tensor(score, dtype=torch.float64, requires_grad=True)

            if args["cuda"]:
                score = score.contiguous().cuda()

            total_loss = score

            # compute gradient and do SGD step
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
    
    def test(self):

        examples = []
        size_train_set = len(os.listdir(self.training_path[0])) 
        if len(self.training_path)>1:
            size_train_set += len(os.listdir(self.training_path[1]))
        instances = [os.path.join(self.testing_path, f) for f in os.listdir(self.testing_path)][:int(0.2 * size_train_set)]

        instance_args = [(instance, self.cut_comp, self.parameter_settings, self.sol_path, self.inputs_type, self.nnet, "test") for instance in instances]
        
        with multiprocessing.Pool(processes=min(multiprocessing.cpu_count(), len(instances))) as pool:
            results = pool.map(self.process_instance, instance_args)
        
        examples = [r for r in results if r is not None]
        if not examples:
            print("Aucun exemple valide trouvé.")
            return
        

        score = shifted_geo_mean(examples)

        return score

    def process_instance(self, instance_args):
        instance_path, cut_comp, parameter_settings, sol_path, inputs_type, nnet, mode = instance_args
        if instance_path.endswith(".lp") or instance_path.endswith(".mps"):
            if mode=="train":
                _, time_or_gap = perform_SCIP_instance(
                    instance_path, 
                    cut_comp,
                    parameter_settings=parameter_settings,
                    sol_path=sol_path,
                    RL=True,
                    inputs_type=inputs_type,
                    nnet=nnet
                )
            elif mode=="test":
                _, time_or_gap = perform_SCIP_instance(
                instance_path,
                cut_comp,
                parameter_settings=parameter_settings,
                sol_path=sol_path,
                is_Test=True,
                RL=True,
                inputs_type=inputs_type,
                nnet=nnet
            )
            return time_or_gap
        else:
            return None
    
    def save_checkpoint(self, folder='', filename='checkpoint', replace=True):
        filepath = os.path.join(folder, filename)
        if replace:
            torch.save({
                'state_dict': self.nnet.state_dict(),
            }, filepath + ".pth.tar")
        else:
            print("WARNING : not saving file", flush=True)

    def load_checkpoint(self, folder='', filename='checkpoint'):
        filepath = os.path.join(folder, filename + ".pth.tar")
        if not os.path.exists(filepath):
            raise ("No model in path {}".format(filepath))
        map_location = None if args["cuda"] else 'cpu'
        checkpoint = torch.load(filepath, map_location=map_location, weights_only=True)
        self.nnet.load_state_dict(checkpoint['state_dict'])