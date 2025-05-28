import os
import shutil
import conf
from data.build_wpms_instances import *

if __name__ == "__main__":
    # Note: The construction of the instance reuses the code by Labassi et al. on Graph Neural Networks.
    # Parameters for instance generation
    n = 50  # Number of training instances
    n_test = 60  # Number of test instances
    whichSet = 'SET2'  # Set identifier, must be set to 'SET2'
    setparam = 100  # Parameter related to "revenues"
    alphaE2 = 0.5  # Probability of building an edge

    problem = "wpsm"

    # Graph parameters for GISP problem representation
    min_n = 60  # Minimum number of nodes in the graph
    max_n = 70  # Maximum number of nodes in the graph
    er_prob = 0.6  # Erdos-Rényi random graph parameter

    """########### SMALL PARAM FOR TESTING ###########
    n=4
    n_test=4
    min_n=30
    max_n=40
    ########### SMALL PARAM FOR TESTING ###########"""

    training_file = f"data/{problem}/train"

    # Directory for training instances
    lp_dir_training = os.path.join(conf.ROOT_DIR, training_file)
    if os.path.exists(lp_dir_training):
        print(f"Cleaning directory: {lp_dir_training}")
        try:
            shutil.rmtree(lp_dir_training)  # Recursively removes a directory and all its contents
        except Exception as e:
            print(f"Error cleaning {lp_dir_training}: {e}")
    else:
        os.makedirs(lp_dir_training)

    # Generate training instances
    #generate_instances(n, whichSet, setparam, alphaE2, min_n, max_n, er_prob, None, lp_dir_training, False) #GISP
    generate_instances(n, min_n, max_n, lp_dir_training, False, er_prob) #WPMS

    test_file = f"data/{problem}/test"

    # Parameters for test instance generation

    # Directory for test instances
    lp_dir_test = os.path.join(conf.ROOT_DIR, test_file)
    if os.path.exists(lp_dir_test):
        print(f"Cleaning directory: {lp_dir_test}")
        try:
            shutil.rmtree(lp_dir_test)  # Recursively removes a directory and all its contents
        except Exception as e:
            print(f"Error cleaning {lp_dir_test}: {e}")
    else:
        os.makedirs(lp_dir_test)

    # Generate test instances"
    #generate_instances(n, whichSet, setparam, alphaE2, min_n, max_n, er_prob, None, lp_dir_test, False) #GISP
    generate_instances(n, min_n, max_n, lp_dir_test, False, er_prob) #WPMS
