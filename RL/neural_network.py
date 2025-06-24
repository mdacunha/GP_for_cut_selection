import os
import numpy as np
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

class nnet(nn.Module):
    def __init__(self, args):
        super(nnet, self).__init__()

        self.args = args

        if args["inputs_type"] == "only_scores":
            input_dim = 1
        elif args["inputs_type"] == "only_features":
            input_dim = 17
        elif args["inputs_type"] == "scores_and_features":
            input_dim = 18
            
        embed_dim=self.args['embed_dim']
        lstm_hidden_dim=self.args['lstm_hidden_dim']

        # 1. Projection des features d'entrée (17D → embed_dim)
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.ReLU = nn.ReLU()

        # 2. Encodeur séquentiel (LSTM unidirectionnel)
        self.lstm = nn.LSTM(embed_dim, lstm_hidden_dim, batch_first=True)

        # 3. MLPs pour prédire µ et log(σ)
        self.mu_head = nn.Linear(lstm_hidden_dim, 1)
        self.log_sigma_head = nn.Linear(lstm_hidden_dim, 1)

    def forward(self, features, sample=True):
        """
        :param features: tensor [N, 17]  (N = nombre de coupes candidates)
        :param sample: bool, si True on échantillonne un ratio k, sinon on renvoie µ
        :return: k ∈ [0,1], µ ∈ ℝ, σ ∈ ℝ⁺
        """
        #N = features.size(0)  # nombre de coupes candidates

        # Étape 1 : Embed les vecteurs 17D → d
        x = self.embedding(features)  # [N, embed_dim]
        x = self.ReLU(x)
        x = x.unsqueeze(0)  # [1, N, embed_dim] → batch_size=1 pour LSTM

        # Étape 2 : Encodeur LSTM → dernier état caché
        _, (h_last, _) = self.lstm(x)  # h_last: [1, hidden_dim]
        h = h_last.squeeze(0)  # [hidden_dim]

        # Étape 3 : MLP pour µ et log(σ)
        mu = self.mu_head(h).squeeze(-1)          # scalaire
        log_sigma = self.log_sigma_head(h).squeeze(-1)  # scalaire
        sigma = torch.exp(log_sigma)

        # Étape 4 : Echantillonnage avec reparametrization trick
        if sample:
            epsilon = torch.randn_like(sigma)
            k_raw = mu + sigma * epsilon
        else:
            k_raw = mu  # pas d'échantillonnage pendant l'inférence

        # Étape 5 : tanh-Gaussian + transformation en [0, 1]
        k_tanh = torch.tanh(k_raw)
        k = 0.5 * (k_tanh + 1.0)

        return k #, mu, sigma
    
    def predict(self, features, mode="train"):
            features = np.concatenate(features, axis=0)
            features = torch.FloatTensor(features.astype(np.float64))
            if self.args["cuda"]: features = features.contiguous().cuda()
            #features = features.view(features.size(0), self.args["num_inputs"])
            if mode=="train":
                self.train()
                k = self.forward(features, sample=True)  # k ∈ [0, 1]
                return k
            elif mode=="test" or mode=="final_test":
                self.eval()
                with torch.no_grad():
                    k = self.forward(features, sample=False)  # k ∈ [0, 1]
                k = k.cpu().numpy()[0]
                if mode=="final_test":
                    json_path = "out.json"
                    with open(json_path, "r") as f:
                        data = json.load(f)
                    last_key = list(data.keys())[-1]
                    data[last_key] = {
                            "k": k
                        }
                    with open(json_path, "w") as f:
                        json.dump(data, f, indent=4)
                return k