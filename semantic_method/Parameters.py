import random

# ==== Données ====
DATA_RNG = 2
MaxDataDim = 130
MaxDataNum = 5000
DataDim = 1
outDataDim = 1
Inf = 1e7
MAXSIZE_SV = 3800

inputs = [0.0] * MaxDataDim
outputs = [0.0] * MaxDataDim
raw_train_input = [0.0] * (MaxDataNum * MaxDataDim)
raw_train_output = [0.0] * (MaxDataNum * MaxDataDim)
train_input = [0.0] * (MaxDataNum * MaxDataDim)
train_output = [0.0] * (MaxDataNum * MaxDataDim)
mean = [0.0] * MaxDataDim
stdu = [0.0] * MaxDataDim
out_mean = 0.0
out_stdu = 0.0
test_input = [0.0] * (MaxDataNum * MaxDataDim)
raw_test_input = [0.0] * (MaxDataNum * MaxDataDim)
test_output = [0.0] * (MaxDataNum * MaxDataDim)

input_num = 0
in_dimen = 0
out_dimen = 0

norm_flag = False

# ==== Opérations ====
MaxChild = 2
MaxReg = 1
registers = [0.0] * MaxReg
NUM_TERM = 3

# ==== Instruction ====
MaxDepth = 5
MUTATION_THRESH = 0.05
CROSSOVER_THRESH = 0.05

# ==== Programme ====
MaxProLen = 5

# ==== Population ====
POPSIZE = 50
evaluation = 0.0
MAXGEN = 100000
MAXEVAL = 1e7

# ==== Semantic library ====
MaxIns = 2000
NUMREF = 5
NumUpdate = 1000
SLTOUSIZE = 20
SPTOUSIZE = 500
DecayF = 0.8
UpdatePeriod = 20
Pthresold = 0.95
DISTRIBUTE = UpdatePeriod * POPSIZE

# ==== Timing (non utilisée telle quelle en Python) ====
dfMinus = 0.0
dfFreq = 0.0
dfTraTime = 0.0
dfTesTime = 0.0

# ==== Fonctions globales ====
def randval(L=0.0, U=1.0):
    return random.uniform(L, U)

def randint(L=0, U=random.randint(1, 1 << 30)):  # U est fixé si non précisé
    return random.randint(L, U)
