import os
from tqdm import tqdm

import numpy as np
from conf import *
import matplotlib.pyplot as plt

import torch
import torch.optim as optim

from RL.neural_network import nnet
from scip_solver import perform_SCIP_instance

args = {
    'lr': 0.00005,
    'dropout': 0.3,
    'epochs': 10,
    'batch_size': 64,
    'cuda': torch.cuda.is_available(),
    'num_channels': 512,
    'num_inputs': 17,
}

class NeuralNetworkWrapper():
    def __init__(self, training_path, testing_path, simulation_folder="", problem="gisp", cut_comp="", parameter_settings=False,
                    saving_folder="", load_checkpoint=False, sol_path=None):
        
        self.nnet = nnet(args)
        self.training_path = training_path
        self.testing_path = testing_path
        self.simulation_folder = simulation_folder
        self.problem = problem
        self.cut_comp = cut_comp
        self.parameter_settings = parameter_settings
        self.saving_folder = saving_folder
        self.sol_path = sol_path
        if load_checkpoint:
            self.load_checkpoint(folder=self.simulation_folder, filename=self.saving_folder)
        if args["cuda"]:
            self.nnet.cuda()     

    def learn(self):
        test_score = np.inf
        for epoch in range(args["epochs"]):
            print('EPOCH ::: ' + str(epoch + 1))
            self.train()
            score = self.test()
            if score < test_score:
                test_score = score
                self.save_checkpoint(folder=self.simulation_folder, filename=self.saving_folder)

    def train(self):

        optimizer = optim.Adam(self.nnet.parameters())

        geo_mean_score = AverageMeter()

        batch_count = 1 #int(len(examples) / args.batch_size)

        t = tqdm(range(batch_count), desc='Training Net')
        for _ in t:
            examples = []
            for instance in os.listdir(self.training_path):
                if instance.endswith(".lp") or instance.endswith(".mps"):
                    instance_path = os.path.join(self.training_path, instance)

                    _, time_or_gap = perform_SCIP_instance(instance_path, self.cut_comp,
                                                            parameter_settings=self.parameter_settings,
                                                            sol_path=self.sol_path, RL=True, nnet=self.nnet,
                                                            )

                    examples.append(time_or_gap)
                    score = shifted_geo_mean(examples)
                    geo_mean_score.update(score)

            
            score = torch.tensor(score, dtype=torch.float64, requires_grad=True)

            if args["cuda"]:
                score = score.contiguous().cuda()

            total_loss = score

            t.set_postfix(Loss=int(score))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
    
    def test(self):
        geo_mean_score = AverageMeter()
        examples = []
        for instance in os.listdir(self.testing_path):
            if instance.endswith(".lp") or instance.endswith(".mps"):
                instance_path = os.path.join(self.testing_path, instance)

                _, time_or_gap = perform_SCIP_instance(instance_path, self.cut_comp,
                                                        parameter_settings=self.parameter_settings,
                                                        sol_path=self.sol_path, is_Test=True, 
                                                        RL=True, nnet=self.nnet                                                        
                                                        )

                examples.append(time_or_gap)
                score = shifted_geo_mean(examples)
                geo_mean_score.update(score)

        return score

    def save_checkpoint(self, folder='', filename='checkpoint', replace=True):
        filepath = os.path.join(folder, filename)
        if replace:
            torch.save({
                'state_dict': self.nnet.state_dict(),
            }, filepath + ".pth.tar")
        else:
            print("WARNING : not saving file", flush=True)

    def load_checkpoint(self, folder='', filename='checkpoint'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise ("No model in path {}".format(filepath))
        map_location = None if args["cuda"] else 'cpu'
        checkpoint = torch.load(filepath, map_location=map_location)
        self.nnet.load_state_dict(checkpoint['state_dict'])

class AverageMeter(object):
    """From https://github.com/pytorch/examples/blob/master/imagenet/main.py"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def __repr__(self):
        return f'{self.avg:.2e}'

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count