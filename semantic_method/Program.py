import numpy as np
import random
from Instructions import Instruction
from Parameters import *
from SemanticLibrary import SemLibrary
from SemanticVector import SV
from global_state import evaluation

class Program:
    def __init__(self):
        self.pro_body = []
        self.sem_con = []
        self.fitness = float('inf')
        self.name = ""
        self.inputS = SV()
        self.targetS = SV()
        self.actual_output = np.zeros(MaxDataNum * outDataDim)
        self.ini_program()
        self.ini_semcon()
        self.get_name()

    def ini_program(self):
        self.pro_body = []
        for _ in range(MaxProLen):
            ins = Instruction()
            ins.root = ins.root.clone()
            self.pro_body.append(ins)

    def get_name(self):
        self.name = "\n".join(ins.name for ins in self.pro_body)

    def ini_semcon(self):
        self.sem_con = [SV() for _ in range(MaxProLen)]
        for i in range(input_num):
            if i * MaxReg < self.targetS.size():
                self.targetS[i * MaxReg] = train_output[i * MaxDataDim]

    def ini_exe(self, index):
        global registers, inputs, outputs
        registers[:] = 0
        for i in range(in_dimen):
            inputs[i] = train_input[index * MaxDataDim + i]
        for i in range(out_dimen):
            outputs[i] = train_output[index * MaxDataDim + i]

    def ini_test_exe(self, index):
        global registers, inputs, outputs
        registers[:] = 0
        for i in range(in_dimen):
            inputs[i] = test_input[index * MaxDataDim + i]
        for i in range(out_dimen):
            outputs[i] = test_output[index * MaxDataDim + i]

    def record_semantic(self, ins_ind, i):
        self.sem_con[ins_ind].setSem(i, registers)

    def get_train_fitness(self):
        self.fitness = 0
        for i in range(input_num):
            for j in range(out_dimen):
                delta = min((train_output[i * MaxDataDim + j] * out_stdu + out_mean) -
                            (self.actual_output[i * outDataDim + j] * out_stdu + out_mean), 1e4)
                self.fitness += delta ** 2
        self.fitness = np.sqrt(self.fitness / input_num)
        return self.fitness

    def get_test_fitness(self):
        self.fitness = 0
        for i in range(input_num):
            for j in range(out_dimen):
                delta = min(test_output[i * MaxDataDim + j] -
                            (self.actual_output[i * outDataDim + j] * out_stdu + out_mean), 1e4)
                self.fitness += delta ** 2
        self.fitness = np.sqrt(self.fitness / input_num)
        return self.fitness

    def get_test_R2(self):
        res, mean, var = 0, 0, 0
        for i in range(input_num):
            for j in range(out_dimen):
                delta = min(test_output[i * MaxDataDim + j] -
                            (self.actual_output[i * outDataDim + j] * out_stdu + out_mean), 1e4)
                res += delta ** 2
        res /= input_num
        mean = np.mean([test_output[i * MaxDataDim] for i in range(input_num)])
        var = np.mean([(test_output[i * MaxDataDim] - mean) ** 2 for i in range(input_num)])
        return 1.0 - res / var

    def program_exe(self):
        global evaluation
        self.actual_output[:] = 0
        for i in range(input_num):
            self.ini_exe(i)
            for ins in self.pro_body:
                ins.eval()
                self.record_semantic(self.pro_body.index(ins), i)
            self.actual_output[i * outDataDim] = registers[0]
        evaluation += self.countProLen()
        self.get_train_fitness()

    def program_test_exe(self):
        self.sem_con = [SV() for _ in range(MaxProLen)]
        self.actual_output[:] = 0
        for i in range(input_num):
            self.ini_test_exe(i)
            for ins in self.pro_body:
                ins.eval()
                self.record_semantic(self.pro_body.index(ins), i)
            self.actual_output[i * outDataDim] = registers[0]
        self.get_test_fitness()

    def countProLen(self):
        return sum(ins.countNode() for ins in self.pro_body)

    def copy(self):
        new_p = Program()
        new_p.pro_body = [ins.copy() for ins in self.pro_body]
        new_p.sem_con = list(self.sem_con)
        new_p.name = self.name
        new_p.fitness = self.fitness
        new_p.inputS = self.inputS
        new_p.targetS = self.targetS
        return new_p

    def __eq__(self, other):
        return self.name == other.name

    def proMutate(self):
        for ins in self.pro_body:
            ins.mutateInstruction()
        self.get_name()

    def proCrossover(self, donatePro):
        for i in range(MaxProLen):
            don_ins = donatePro.pro_body[random.randint(0, MaxProLen - 1)]
            self.pro_body[i].crossOver(don_ins)
        self.get_name()

    def proDEMutate(self, target, r2):
        for i in range(MaxProLen):
            tar_ins = target.pro_body[i]
            r2_ins = r2.pro_body[i]
            self.pro_body[i].DE_mutate(tar_ins.root, r2_ins.root)
        self.get_name()

    def MutateAndDivide(self, sem_lib: SemLibrary, inS: SV, desS: SV, head=0, tail=None, STEP=0):
        if tail is None:
            tail = MaxProLen - 1
        if STEP == 0:
            STEP = random.randint(1, MaxProLen)
        InputSlot = max(head, random.randint(tail - STEP, tail - 1))
        MutateSlot = random.randint(InputSlot + 1, tail)

        X_star = inS if InputSlot == head else self.sem_con[InputSlot]

        trail, DI, EO = sem_lib.selectInstr(X_star, desS)
        self.pro_body[MutateSlot] = trail

        if MutateSlot < tail:
            self.MutateAndDivide(sem_lib, EO, desS, MutateSlot, tail, STEP)
        if InputSlot < MutateSlot - 1:
            DI = self.MutateAndDivide(sem_lib, X_star, DI, InputSlot, MutateSlot - 1, STEP)
        if head < InputSlot:
            DI = self.MutateAndDivide(sem_lib, inS, DI, head, InputSlot, STEP)

        return DI
