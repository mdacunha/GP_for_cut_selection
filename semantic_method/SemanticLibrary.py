import numpy as np
import random
from Instructions import Instruction
from SemanticVector import SV

# Constants (doivent être cohérents avec parameters.py)
NUMREF = 5
MaxDataDim = 130
MaxDataNum = 5000
MaxReg = 1
MaxIns = 2000
SLTOUSIZE = 20
SPTOUSIZE = 500
DecayF = 0.8
UpdatePeriod = 20
Pthresold = 0.95
NumUpdate = 1000
DISTRIBUTE = UpdatePeriod * 50  # POPSIZE

Inf = 1e7
input_num = 0  # à ajuster dynamiquement
in_dimen = 0   # à ajuster dynamiquement

# Données globales simulées (à relier à votre pipeline réel)
train_input = np.zeros((MaxDataNum, MaxDataDim))
registers = np.zeros(MaxReg)

def randval(L=0, U=1):
    return random.uniform(L, U)

class SVSet:
    def __init__(self):
        self.sem_vec = [SV() for _ in range(NUMREF)]

    def __eq__(self, other):
        return all(self.sem_vec[i] == other.sem_vec[i] for i in range(NUMREF))

    def __getitem__(self, index):
        if index >= NUMREF:
            raise IndexError("SVSet index out of range")
        return self.sem_vec[index]

    def __setitem__(self, index, value):
        self.sem_vec[index] = value

    def copy(self):
        new_set = SVSet()
        for i in range(NUMREF):
            new_set[i] = self.sem_vec[i].copy()
        return new_set


class SemLibrary:
    def __init__(self):
        self.inRefSV = [SV() for _ in range(NUMREF)]
        for sv in self.inRefSV:
            sv.rand_set_sem(-2, 2)

        self.InsList = []
        self.outRefSV = []
        self.occurFre = []

        for _ in range(MaxIns):
            ins = Instruction()
            if ins not in self.InsList:
                out_sv = self.eval_instr(ins)
                if out_sv not in self.outRefSV:
                    self.InsList.append(ins.copy())
                    self.outRefSV.append(out_sv)
                    self.occurFre.append(0)

    def eval_instr(self, ins):
        tmpout = SVSet()
        for n in range(NUMREF):
            for i in range(min(input_num, SV.maxsize)):
                for j in range(in_dimen):
                    ins.inputs[j] = train_input[i, j]
                for j in range(MaxReg):
                    registers[j] = self.inRefSV[n][i * MaxReg + j]
                ins.eval()
                for j in range(MaxReg):
                    tmpout[n][i * MaxReg + j] = registers[j]
        return tmpout

    def update_library(self):
        tmp_ins_list = []
        elim_list = []

        for _ in range(NumUpdate):
            p = self.select_top3()
            trail = self.InsList[p[0]].copy()
            trail.de_mutate(self.InsList[p[1]].root, self.InsList[p[2]].root)
            tmp_ins_list.append(trail)
            e = self.select_elim(p)
            elim_list.append(e)

        for t in range(NumUpdate):
            trail = tmp_ins_list[t]
            if trail not in self.InsList:
                out_sv = self.eval_instr(trail)
                if out_sv not in self.outRefSV:
                    if len(self.InsList) < MaxIns:
                        self.InsList.append(trail.copy())
                        self.outRefSV.append(out_sv)
                        self.occurFre.append(0)
                    else:
                        e = elim_list[t]
                        self.InsList[e] = trail.copy()
                        self.outRefSV[e] = out_sv
                        self.occurFre[e] = 0

        self.occurFre = [f * DecayF for f in self.occurFre]

    def select_top3(self):
        selected = []
        while len(selected) < 3:
            best = -1
            best_score = -1
            for _ in range(SLTOUSIZE):
                idx = random.randint(0, len(self.InsList) - 1)
                if idx in selected:
                    continue
                prob = min(self.occurFre[idx] / DISTRIBUTE, Pthresold)
                if best == -1 or (self.occurFre[idx] > best_score and randval() <= prob):
                    best = idx
                    best_score = self.occurFre[idx]
            selected.append(best)
        return selected

    def select_elim(self, avoid):
        worst = -1
        worst_score = float('inf')
        for _ in range(SLTOUSIZE):
            idx = random.randint(0, len(self.InsList) - 1)
            if idx in avoid or idx in avoid:
                continue
            prob = min(self.occurFre[idx] / DISTRIBUTE, Pthresold)
            if worst == -1 or (self.occurFre[idx] < worst_score and randval() > prob):
                worst = idx
                worst_score = self.occurFre[idx]
        return worst

    def estimateX(self, insInd, Y_star):
        recordSV = self.outRefSV[insInd]
        index0 = min(range(NUMREF), key=lambda i: recordSV[i].vec_diff(Y_star))
        if recordSV[index0].vec_diff(Y_star) == 0:
            return self.inRefSV[index0]

        readRegs = self.InsList[insInd].get_read_regs()
        X_star = SV()

        for k in range(X_star.size()):
            if (k % MaxReg) not in readRegs:
                X_star[k] = Y_star[k]
            else:
                x0k = self.inRefSV[index0][k]
                dey = Y_star[k] - recordSV[index0][k]
                sumX = sum(self.inRefSV[i][k] - x0k for i in range(NUMREF) if i != index0)
                sumY = sum(recordSV[i][k] - recordSV[index0][k] for i in range(NUMREF) if i != index0)
                if dey == 0 or sumX == 0:
                    X_star[k] = x0k
                elif sumY != 0:
                    X_star[k] = x0k + dey * (sumX / sumY)
                else:
                    X_star[k] = Inf
                if abs(X_star[k]) > Inf:
                    X_star[k] = Inf
        return X_star

    def estimateY(self, insInd, X_star):
        recordSV = self.outRefSV[insInd]
        index0 = min(range(NUMREF), key=lambda i: self.inRefSV[i].vec_diff(X_star))
        if self.inRefSV[index0].vec_diff(X_star) == 0:
            return recordSV[index0]

        writeRegs = self.InsList[insInd].get_write_regs()
        Y_star = SV()

        for k in range(Y_star.size()):
            if (k % MaxReg) not in writeRegs:
                Y_star[k] = X_star[k]
            else:
                y0k = recordSV[index0][k]
                dex = X_star[k] - self.inRefSV[index0][k]
                sumX = sum(self.inRefSV[i][k] - self.inRefSV[index0][k] for i in range(NUMREF) if i != index0)
                sumY = sum(recordSV[i][k] - y0k for i in range(NUMREF) if i != index0)
                if dex == 0 or sumY == 0:
                    Y_star[k] = y0k
                elif sumX != 0:
                    Y_star[k] = y0k + dex * (sumY / sumX)
                else:
                    Y_star[k] = Inf
                if abs(Y_star[k]) > Inf:
                    Y_star[k] = Inf
        return Y_star

    def select_instr(self, inputS, Sdes):
        tci = -1
        min_diff = float('inf')
        for _ in range(SPTOUSIZE):
            ci = random.randint(0, len(self.InsList) - 1)
            prob = min(self.occurFre[ci] / DISTRIBUTE, Pthresold)
            if tci == -1 or randval() > prob:
                diff = self.estimateY(ci, inputS).vec_diff(Sdes)
                if diff < min_diff:
                    min_diff = diff
                    tci = ci

        Tinstr = self.InsList[tci].copy()
        DI = self.estimateX(tci, Sdes)
        EO = self.estimateY(tci, inputS)
        self.occurFre[tci] += 1

        return Tinstr, DI, EO
