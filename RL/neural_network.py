import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class nnet2(nn.Module):
    def __init__(self, args, parallel_filtering=False, outputs=100):
        super(nnet2, self).__init__()

        self.args = args
        self.outputs = outputs

        if self.args["inputs_type"] == "only_scores":
            input_dim = 1
        elif self.args["inputs_type"] == "only_features":
            input_dim = 17
        elif self.args["inputs_type"] == "scores_and_features":
            input_dim = 18

        if parallel_filtering: 
            input_dim+=1

        self.input_dim = input_dim
            
        lstm_hidden_dim = self.args['lstm_hidden_dim']

        self.ReLU = nn.ReLU()

        self.lstm = nn.LSTM(input_dim, lstm_hidden_dim, batch_first=True)
        self.fc = nn.Linear(lstm_hidden_dim, outputs)

    def forward(self, x):
        """
        x: (B, L, 2)
        """
        x = x.unsqueeze(0)
        #assert L <= self.k_max, f"Input length {L} exceeds k_max {self.k_max}"
        
        _, (h_n, _) = self.lstm(x)
        h_final = h_n[-1]
        logits = self.fc(h_final)

        # Crée la distribution catégorielle directement sur les logits
        dist = Categorical(logits=logits)
        return dist
    
    def predict(self, features, mode="test"):
            features = np.stack(features)
            features = torch.tensor(features, dtype=torch.float32)
            features = features.view(-1, self.input_dim)
            if self.args["cuda"]: features = features.contiguous().cuda()
            #features = features.view(features.size(0), self.args["num_inputs"])
            if mode=="train":
                self.train()
                dist = self.forward(features)  # k ∈ [0, 1]
                return dist
            elif mode=="test" or mode=="final_test":
                self.eval()
                with torch.no_grad():
                    dist = self.forward(features)  # k ∈ [0, 1]
                    k = torch.argmax(dist.probs).cpu().item() + 1  # échantillonne une action
                return k
            
class nnet1(nn.Module):
    def __init__(self, args, parallel_filtering=False, outputs=100):
        super(nnet1, self).__init__()

        self.args = args

        if self.args["inputs_type"] == "only_scores":
            input_dim = 1
        elif self.args["inputs_type"] == "only_features":
            input_dim = 17
        elif self.args["inputs_type"] == "scores_and_features":
            input_dim = 18

        if parallel_filtering: 
            input_dim+=1

        self.input_dim = input_dim
            
        lstm_hidden_dim= self.args['lstm_hidden_dim']
        mlp1_dim = self.args['mlp1_dim']
        mlp2_dim = self.args['mlp2_dim']

        self.dropout = nn.Dropout(self.args['dropout'])
        self.ln = nn.LayerNorm(lstm_hidden_dim)

        self.lstm = nn.LSTM(input_dim, lstm_hidden_dim, batch_first=True)
        self.fc = torch.nn.Sequential(
            nn.Linear(lstm_hidden_dim, mlp1_dim),
            nn.ReLU(),
            self.dropout,
            nn.Linear(mlp1_dim, mlp2_dim),
            nn.ReLU(),
            self.dropout,
            nn.Linear(mlp2_dim, outputs)
        )

        self.attention = Attention(lstm_hidden_dim)
        

    def forward(self, x):
        """
        x: (L, 2)
        """        
        x = x.unsqueeze(0)
        lstm_out, _ = self.lstm(x)
        #h_final = h_n[-1]
        out = self.ln(lstm_out)
        context, attn_weights = self.attention(out)
        x = self.dropout(context)
        logits = self.fc(x)


        # Crée la distribution catégorielle directement sur les logits
        dist = Categorical(logits=logits)
        return dist
    
    def predict(self, features, mode="test"):
            features = np.stack(features)
            features = torch.tensor(features, dtype=torch.float32)
            features = features.view(-1, self.input_dim)
            if self.args["cuda"]: features = features.contiguous().cuda()
            #features = features.view(features.size(0), self.args["num_inputs"])
            self.mode=mode
            if mode=="train":
                self.train()
                dist = self.forward(features)  # k ∈ [0, 1]
                return dist
            elif mode=="test" or mode=="final_test":
                self.eval()
                with torch.no_grad():
                    dist = self.forward(features)  # k ∈ [0, 1]
                    k = torch.argmax(dist.probs).cpu().item() + 1  # échantillonne une action
                return k

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, lstm_outputs):  # (batch, seq_len, hidden_size)
        # Score pour chaque timestep
        attn_scores = self.attn(lstm_outputs).squeeze(-1)  # (batch, seq_len)

        # Normalisation softmax
        attn_weights = torch.softmax(attn_scores, dim=1)  # (batch, seq_len)

        # Calcul du contexte (pondération des outputs par attention)
        context = torch.sum(lstm_outputs * attn_weights.unsqueeze(-1), dim=1)  # (batch, hidden_size)

        return context, attn_weights  # On peut retourner les poids pour visualisation

class nnet0(nn.Module):
    def __init__(self, args, parallel_filtering=False):
        super(nnet0, self).__init__()

        self.args = args

        if self.args["inputs_type"] == "only_scores":
            input_dim = 1
        elif self.args["inputs_type"] == "only_features":
            input_dim = 17
        elif self.args["inputs_type"] == "scores_and_features":
            input_dim = 18
            
        if parallel_filtering: 
            input_dim+=1
        
        self.input_dim = input_dim

        embed_dim=self.args['embed_dim']
        lstm_hidden_dim=self.args['lstm_hidden_dim']

        # 1. Projection des features d'entrée (1D → embed_dim)
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.ReLU = nn.ReLU()

        # 2. Encodeur séquentiel (LSTM unidirectionnel)
        self.lstm = nn.LSTM(input_dim, lstm_hidden_dim, batch_first=True)

        # 3. MLPs pour prédire µ et log(σ)
        self.mu_head = nn.Linear(lstm_hidden_dim, 1)
        self.log_sigma_head = nn.Linear(lstm_hidden_dim, 1)

    def forward(self, features, sample=True):
        """
        :param features: tensor [N, 2]  (N = nombre de coupes candidates)
        :param sample: bool, si True on échantillonne un ratio k, sinon on renvoie µ
        :return: k ∈ [0,1], µ ∈ ℝ, σ ∈ ℝ⁺
        """
        #N = features.size(0)  # nombre de coupes candidates

        # Étape 1 : Embed les vecteurs 17D → d
        #x = self.embedding(features) # [N, embed_dim]
        #x = self.ReLU(x)
        x = features.unsqueeze(0)  # [1, N, embed_dim] → batch_size=1 pour LSTM

        # Étape 2 : Encodeur LSTM → dernier état caché
        _, (h_last, _) = self.lstm(x)  # h_last: [1, hidden_dim]
        h = h_last.squeeze(0)  # [hidden_dim]

        # Étape 3 : MLP pour µ et log(σ)
        mu = self.mu_head(h).squeeze(-1)          # scalaire
        log_sigma = self.log_sigma_head(h).squeeze(-1)  # scalaire
        sigma = torch.exp(log_sigma).clamp(min=1e-4, max=1.0)

        # Étape 4 : Echantillonnage avec reparametrization trick
        if sample:
            epsilon = torch.randn_like(sigma)
            k_raw = mu + sigma * epsilon
        else:
            k_raw = mu  # pas d'échantillonnage pendant l'inférence

        # Étape 5 : tanh-Gaussian + transformation en [0, 1]
        k_tanh = torch.tanh(k_raw)
        k = 0.5 * (k_tanh + 1.0)

        return k, k_raw, mu, sigma
    
    def predict(self, features, mode="train"):
            features = np.stack(features)
            features = torch.tensor(features, dtype=torch.float32)
            features = features.view(-1, self.input_dim)
            if self.args["cuda"]: features = features.contiguous().cuda()
            if mode=="train":
                self.train()
                k, k_raw, mu, sigma = self.forward(features, sample=True)  # k ∈ [0, 1]
                return k, k_raw, mu, sigma
            elif mode=="test" or mode=="final_test":
                self.eval()
                with torch.no_grad():
                    k, _, _, _ = self.forward(features, sample=False)  # k ∈ [0, 1]
                k = k.cpu().item()
                return k
            
