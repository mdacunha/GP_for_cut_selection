import os
import shutil
import conf
from data.build_gisp_instances import *

if __name__ == "__main__":
    # Note: The construction of the instance reuses the code by Labassi et al. on Graph Neural Networks.
    # Parameters for instance generation
    n = 50  # Number of training instances
    whichSet = 'SET2'  # Set identifier, must be set to 'SET2'
    setparam = 100  # Parameter related to "revenues"
    alphaE2 = 0.5  # Probability of building an edge

    # Graph parameters for GISP problem representation
    min_n = 60  # Minimum number of nodes in the graph
    max_n = 70  # Maximum number of nodes in the graph
    er_prob = 0.6  # Erdos-Rényi random graph parameter

    training_file = "data/gisp/train_for_jupyter"

    # Directory for training instances
    lp_dir_training = os.path.join(conf.ROOT_DIR, training_file)
    if os.path.exists(lp_dir_training):
        print(f"Cleaning directory: {lp_dir_training}")
        try:
            shutil.rmtree(lp_dir_training)  # Recursively removes a directory and all its contents
        except Exception as e:
            print(f"Error cleaning {lp_dir_training}: {e}")
    else:
        os.mkdir(lp_dir_training)

    # Generate training instances
    generate_instances(n, whichSet, setparam, alphaE2, min_n, max_n, er_prob, None, lp_dir_training, False)

    test_file = "data/gisp/test_for_jupyter"

    # Parameters for test instance generation
    n = 60  # Number of test instances, increased by 10 in case they are failures in the evaluation through all the baselines

    # Directory for test instances
    lp_dir_test = os.path.join(conf.ROOT_DIR, test_file)
    if os.path.exists(lp_dir_test):
        print(f"Cleaning directory: {lp_dir_test}")
        try:
            shutil.rmtree(lp_dir_test)  # Recursively removes a directory and all its contents
        except Exception as e:
            print(f"Error cleaning {lp_dir_test}: {e}")
    else:
        os.mkdir(lp_dir_test)

    # Generate test instances
    generate_instances(n, whichSet, setparam, alphaE2, min_n, max_n, er_prob, None, lp_dir_test, False)
