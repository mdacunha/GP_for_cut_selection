import numpy as np
import os

# Constantes (à adapter si besoin)
MaxDataNum = 10000
MaxDataDim = 100
input_num = 0
in_dimen = 0
out_dimen = 0
norm_flag = True

# Données globales
raw_train_input = np.zeros((MaxDataNum, MaxDataDim))
raw_train_output = np.zeros((MaxDataNum, MaxDataDim))
train_input = np.zeros((MaxDataNum, MaxDataDim))
train_output = np.zeros((MaxDataNum, MaxDataDim))

raw_test_input = np.zeros((MaxDataNum, MaxDataDim))
test_input = np.zeros((MaxDataNum, MaxDataDim))
test_output = np.zeros((MaxDataNum, MaxDataDim))

mean = np.zeros(MaxDataDim)
stdu = np.ones(MaxDataDim)
out_mean = 0
out_stdu = 1

# Génère des valeurs entre deux bornes
def randval(low, high):
    return np.random.uniform(low, high)

# Génère des données artificielles
def genData():
    global train_input, train_output
    for i in range(MaxDataNum):
        d = randval(0, 2)
        train_input[i, 0] = d
        train_output[i, 0] = np.log(d * d + 1) + np.log(d + 1)

# Applique une normalisation standard
def StandardScaler_transform(raw, des, length, mean_p, stdu_p):
    for j in range(length):
        des[:, j] = (raw[:, j] - mean_p[j]) / stdu_p[j]

# Calcule et applique une normalisation standard
def StandardScaler_fit_transform(raw, des, length, mean_p, stdu_p):
    for j in range(length):
        mean_p[j] = np.mean(raw[:, j])
        stdu_p[j] = np.std(raw[:, j])
    StandardScaler_transform(raw, des, length, mean_p, stdu_p)

# Lecture des données d'entraînement
def readTrainData(run, job):
    global input_num, in_dimen, out_dimen, out_mean, out_stdu
    filename = f"/SLGP_exp/SLGP_for_SR_CYB/F{run}_{job}_training_data.txt"
    print(filename)
    
    with open(filename, "r") as f:
        row_num, col_num = map(int, f.readline().split())
        input_num = min(row_num, MaxDataNum)
        in_dimen = col_num
        out_dimen = 1
        
        for i in range(row_num):
            line = f.readline().split()
            raw_train_input[i, :col_num] = list(map(float, line[:col_num]))
            raw_train_output[i, 0] = float(line[col_num])

    if norm_flag:
        StandardScaler_fit_transform(raw_train_input, train_input, in_dimen, mean, stdu)
        StandardScaler_fit_transform(raw_train_output, train_output, 1, np.array([out_mean]), np.array([out_stdu]))
        out_mean = mean[0]
        out_stdu = stdu[0]
    else:
        train_input[:input_num, :in_dimen] = raw_train_input[:input_num, :in_dimen]
        train_output[:input_num, 0] = raw_train_output[:input_num, 0]
        out_mean, out_stdu = 0, 1

# Lecture des données de test
def readTestData(run, job):
    global input_num, in_dimen, out_dimen, out_mean, out_stdu
    filename = f"/SLGP_exp/SLGP_for_SR_CYB/F{run}_{job}_testing_data.txt"
    print(filename)

    with open(filename, "r") as f:
        row_num, col_num = map(int, f.readline().split())
        input_num = min(row_num, MaxDataNum)
        in_dimen = col_num
        out_dimen = 1
        
        for i in range(row_num):
            line = f.readline().split()
            raw_test_input[i, :col_num] = list(map(float, line[:col_num]))
            test_output[i, 0] = float(line[col_num])

    if norm_flag:
        StandardScaler_transform(raw_test_input, test_input, in_dimen, mean, stdu)
    else:
        test_input[:input_num, :in_dimen] = raw_test_input[:input_num, :in_dimen]
        out_mean, out_stdu = 0, 1
