import os
from tqdm import tqdm

import numpy as np
from conf import *

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR

import torch.multiprocessing as mp
from torch.distributions import Normal

from scip_solver import perform_SCIP_instance
from RL.arguments import args

import multiprocessing
from RL.neural_network import nnet0, nnet1, nnet2

class NeuralNetworkWrapper():
    def __init__(self, training_path="", testing_path="", higher_simulation_folder="", problem="gisp", cut_comp="", 
                 parameter_settings=False, saving_folder="", load_checkpoint=False, inputs_type="", sol_path=None, 
                 parallel=False, glob_model=None, best_score=None, exp=0, parallel_filtering=False):
        
        self.device = torch.device("cuda" if args["cuda"] else "cpu")
        self.exp = int(exp)
        self.parallel_filtering = parallel_filtering
        self.set_nnet(args, inputs_type, glob_model)
        self.training_path = training_path
        self.testing_path = testing_path
        self.higher_simulation_folder = higher_simulation_folder
        self.problem = problem
        self.cut_comp = cut_comp
        self.parameter_settings = parameter_settings
        self.saving_folder = saving_folder
        self.sol_path = sol_path
        self.if_load_checkpoint = load_checkpoint
        if self.if_load_checkpoint:
            self.load_checkpoint(folder=self.higher_simulation_folder, filename=self.saving_folder)
        if args["cuda"]:
            self.nnet.cuda()   
        self.parallel = parallel  
        self.best_score = best_score


    def set_nnet(self, args, inputs_type, model):
        if model is not None:
            self.nnet = model
        else:
            self.new_args = args
            self.inputs_type = inputs_type
            self.new_args.update({
                    'inputs_type': self.inputs_type
                })
            if self.exp==0:
                self.nnet = nnet0(self.new_args, self.parallel_filtering)
            elif self.exp==1:
                self.nnet = nnet1(self.new_args, self.parallel_filtering)
            elif self.exp==2:
                self.nnet = nnet2(self.new_args, self.parallel_filtering)

    def learn(self):
        #mp.set_start_method('spawn', force=True)
        if self.if_load_checkpoint and (self.best_score is not None):
            best_test_score = self.best_score
        else:
            best_test_score = np.inf
        instances = [os.path.join(self.training_path[0], f) for f in os.listdir(self.training_path[0])]
        if len(self.training_path)>1:
            instances += [os.path.join(self.training_path[1], f) for f in os.listdir(self.training_path[1])]

        n = len(instances)
        #self.baselines = {i: 0.0 for i in range(1, n + 1)}

        train_instances = instances[:int(0.8 * n)]  # Utiliser 80% des instances pour l'entraînement
        test_instances = instances[int(0.8 * len(instances)):]  # Utiliser 20% des instances pour les tests
        batch_count = int(len(train_instances) / args['batch_size']) + (len(train_instances) % args['batch_size'] != 0)

        optimizer = optim.Adam(self.nnet.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])

        scheduler = OneCycleLR(optimizer, 
                       max_lr=args['lr'],
                       steps_per_epoch=batch_count, 
                       epochs=args['epochs'])
        #L = []
        for epoch in range(args["epochs"]):
            print('EPOCH ' + str(epoch + 1), flush=True)
            if self.parallel:
                print("Training the neural network in parallel...", flush=True)
                self.parallel_train(train_instances)
                test_score = self.parallel_test(test_instances)
            else:
                print("Training the neural network...", flush=True)
                self.train(train_instances, batch_count, optimizer, scheduler)
                #print("Training solving time :", float(train_score[0]), flush=True)
                test_score = self.test(test_instances)
                #L.append([train_score, test_score])
            print("Global solving time for test instances :", test_score, flush=True)
            if test_score < best_test_score:
                best_test_score = test_score
                print("Saving model...", flush=True)
                self.save_checkpoint(folder=self.higher_simulation_folder, filename=self.saving_folder)
            for param_group in optimizer.param_groups:
                print(f"LR actuel : {param_group['lr']:.6f}")
        #plot_evolution(L)
        return best_test_score
    
    def parallel_train(self, instances):
        self.nnet.share_memory()  # important pour le modèle global
        optimizer = optim.Adam(self.nnet.parameters(), lr=args['lr'])

        instances = instances[:int(0.8 * len(instances))]  # Utiliser 80% des instances pour l'entraînement

        num_processes = min(mp.cpu_count(), len(instances))

        processes = []
        for rank in range(num_processes):
            p = mp.Process(target=self.train_worker, args=(rank, instances[rank], self.nnet, optimizer))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
        import gc
        gc.collect()

    def train_worker(self, rank, instance, shared_model, optimizer):
        torch.manual_seed(rank)  # pour diversité

        local_model = nnet(self.new_args)
        if self.new_args["cuda"]:
            local_model.cuda()  
        local_model.load_state_dict(shared_model.state_dict())


        instance_args = (instance, self.cut_comp, self.parameter_settings, self.sol_path, self.inputs_type, local_model, "train")
        r, k_list = self.process_instance(instance_args)

        if r is not None and k_list:
            loss = r * torch.sum(torch.log(torch.stack([k + 1e-8 for k in k_list])))
            optimizer.zero_grad()
            loss.backward()

            for param, shared_param in zip(local_model.parameters(), shared_model.parameters()):
                if shared_param.grad is None:
                    shared_param.grad = param.grad.clone()
                else:
                    shared_param.grad += param.grad

        # Synchronisation centralisée : update global
        optimizer.step()
        shared_model.zero_grad()


    def train(self, instances, batch_count, optimizer, scheduler):
        
        t = range(batch_count)
        for i in t:
            results = []
            for instance in instances[i * args['batch_size'] : min((i+1) * args['batch_size'], len(instances))]:
                instance_args = (instance, self.cut_comp, self.parameter_settings, self.sol_path, self.inputs_type, self.nnet, "train") 


                reward, log_sample_list = self.process_instance(instance_args)
                results.append((reward, log_sample_list))

            """for j in range(i * args['batch_size'], min((i+1) * args['batch_size'], len(instances))):
                b = self.baselines[j+1]
                #print(len(results), j-i * args['batch_size'])
                r = results[j-i * args['batch_size']][0]
                temp = list(results[j - i * args['batch_size']])
                temp[0] = r - b
                results[j - i * args['batch_size']] = tuple(temp)
                new_b = (1 - args['alpha']) * b + args['alpha'] * r
                self.baselines[j+1] = new_b"""

            if self.exp==0:
                losses = []
                for result in results:
                    reward, outputs = result
                    reward = torch.tensor(reward, dtype=torch.float32, device=self.device)
                    single_instance_loss = [-Normal(mu, sigma).log_prob(k_raw) for (k_raw, mu, sigma) in outputs]
                    losses.append(reward * torch.stack(single_instance_loss).mean())
            else:
                losses = [-reward * torch.stack(log_sample_list).sum() for (reward, log_sample_list) in results]
            
            loss = torch.stack(losses).mean()

            #score = shifted_geo_mean([r for (r, _) in results])
        
            if args["cuda"]:
                loss = loss.contiguous().cuda()

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(self.nnet.parameters(), max_norm=5.0)
            optimizer.step()
            scheduler.step()

            return np.mean([r for (r, _) in results if r is not None]), 
        
        """result = [k.item() for k in results[0][1]]
        print("k_list :", result)"""
    
    def parallel_test(self, instances):

        examples = []
        instances = instances[int(0.8 * len(instances)):]  # Utiliser 80% des instances pour l'entraînement

        instance_args = [(instance, self.cut_comp, self.parameter_settings, self.sol_path, self.inputs_type, self.nnet, "test") for instance in instances]
        
        with multiprocessing.Pool(processes=min(multiprocessing.cpu_count(), len(instances))) as pool:
            results = pool.map(self.process_instance, instance_args)
        
        examples = [r for (r, _) in results if r is not None]
        if not examples:
            print("Aucun exemple valide trouvé.")
            return
        

        score = shifted_geo_mean(examples)

        return score
    
    def test(self, instances):

        examples = []

        instance_args = [(instance, self.cut_comp, self.parameter_settings, self.sol_path, self.inputs_type, self.nnet, "test") for instance in instances]
        results = []
        for instance_arg in instance_args:
            reward, log_sample_list = self.process_instance(instance_arg)
            results.append((reward, log_sample_list))
        
        examples = [r for (r, _) in results if r is not None]
        if not examples:
            print("Aucun exemple valide trouvé.")
            return

        score = shifted_geo_mean(examples)

        return score

    def process_instance(self, instance_args):
        instance_path, cut_comp, parameter_settings, sol_path, inputs_type, nnet, mode = instance_args
        if instance_path.endswith(".lp") or instance_path.endswith(".mps"):
            if mode=="train":
                _, time_or_gap, k_list, t = perform_SCIP_instance(
                    instance_path, 
                    cut_comp,
                    parameter_settings=parameter_settings,
                    sol_path=sol_path,
                    RL=True,
                    inputs_type=inputs_type,
                    nnet=nnet,
                    exp=self.exp,
                    parallel_filtering=self.parallel_filtering
                )
            elif mode=="test":
                _, time_or_gap, k_list, t = perform_SCIP_instance(
                instance_path,
                cut_comp,
                parameter_settings=parameter_settings,
                sol_path=sol_path,
                is_Test=True,
                RL=True,
                inputs_type=inputs_type,
                nnet=nnet,
                exp=self.exp,
                parallel_filtering=self.parallel_filtering
            )
            return time_or_gap, k_list
        else:
            return None, None
    
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

import matplotlib.pyplot as plt

def plot_evolution(values):
    """
    Affiche l'évolution (courbes) des deux séries de valeurs contenues dans `values`.
    Chaque élément de `values` est une paire (val1, val2).
    """
    # Séparer les deux séries
    list1 = [v[0] for v in values]
    list2 = [v[1] for v in values]

    # Créer la figure
    plt.figure(figsize=(10, 5))
    plt.plot(list1, label='train', color='blue', marker='o')
    plt.plot(list2, label='test', color='orange', marker='x')
    plt.title('Évolution des deux séries de valeurs')
    plt.xlabel('Itération')
    plt.ylabel('Valeur')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
