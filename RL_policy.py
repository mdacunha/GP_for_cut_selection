import torch
import torch.nn as nn
import torch.nn.functional as F

class HighLevelPolicy(nn.Module):
    def __init__(self, input_dim=13, embed_dim=64, lstm_hidden_dim=128):
        super(HighLevelPolicy, self).__init__()
        
        # 1. Projection des features d'entrée (13D → embed_dim)
        self.embedding = nn.Linear(input_dim, embed_dim)

        # 2. Encodeur séquentiel (LSTM unidirectionnel)
        self.lstm = nn.LSTM(embed_dim, lstm_hidden_dim, batch_first=True)

        # 3. MLPs pour prédire µ et log(σ)
        self.mu_head = nn.Linear(lstm_hidden_dim, 1)
        self.log_sigma_head = nn.Linear(lstm_hidden_dim, 1)

    def forward(self, features, sample=True):
        """
        :param features: tensor [N, 13]  (N = nombre de coupes candidates)
        :param sample: bool, si True on échantillonne un ratio k, sinon on renvoie µ
        :return: k ∈ [0,1], µ ∈ ℝ, σ ∈ ℝ⁺
        """
        N = features.size(0)  # nombre de coupes candidates

        # Étape 1 : Embed les vecteurs 13D → d
        x = self.embedding(features)  # [N, embed_dim]
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

        return k, mu, sigma
