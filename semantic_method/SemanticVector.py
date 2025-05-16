import numpy as np
import random

# Dépendances globales / constantes (à ajuster selon votre projet)
MAXSIZE_SV = 5000  # à adapter si défini dans Parameters.h
MaxReg = 1         # idem
input_num = 0      # doit être défini dynamiquement dans le projet
Inf = 1e7          # pour les erreurs de dimension

class SV:
    maxsize = MAXSIZE_SV

    def __init__(self):
        size = min(input_num, SV.maxsize) * MaxReg
        self.content = np.zeros(size, dtype=np.float64)

    def vec_diff(self, sv2):
        if sv2.size() != self.size():
            print("dimension inconsistent in vec_diff")
            return Inf
        return np.abs(self.content - sv2.content).sum()

    def set_sem(self, index, sv2, length=MaxReg):
        """
        index : int — index d'entrée
        sv2   : array-like — valeurs à insérer
        length: int — nombre de valeurs (par défaut = MaxReg)
        """
        start = index * MaxReg
        if start + length <= self.size():
            self.content[start:start + length] = sv2[:length]

    def rand_set_sem(self, lowb=-2.0, upb=2.0):
        self.content = np.random.uniform(low=lowb, high=upb, size=self.size())

    def __getitem__(self, i):
        return self.content[i]

    def __setitem__(self, i, value):
        self.content[i] = value

    def __eq__(self, other):
        if not isinstance(other, SV):
            return False
        return np.allclose(self.content, other.content)

    def __len__(self):
        return self.size()

    def size(self):
        return self.content.size

    def copy(self):
        new_sv = SV()
        new_sv.content = self.content.copy()
        return new_sv
