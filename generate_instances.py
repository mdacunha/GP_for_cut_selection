import os
import shutil
import conf
from data import build_gisp_instances, build_wpsm_instances, build_fcmcnf_instances

if __name__ == "__main__":
    # Note: The construction of the instance reuses the code by Labassi et al. on Graph Neural Networks.
    # Parameters for instance generation
    problems = ["gisp"]#, "wpsm", "fcmcnf"]  # List of problems to generate instances for
    extend_training_instances = False

    for problem in problems:
        if problem=="gisp" or problem=="wpsm":
            min_n_train = 60  # Minimum number of nodes in the graph
            max_n_train = 70  # Maximum number of nodes in the graph
            
            min_n_test = 120  # Minimum number of nodes in the graph
            max_n_test = 140  # Maximum number of nodes in the graph
            er_prob = 0.6  # Erdos-Rényi random graph parameter
        elif problem=="fcmcnf":
            er_prob = 0.3
            ntrain =15
            ntest = 20
            n_commodities_train = int(1.5*ntrain)
            n_commodities_test = int(1.5*ntest)

        n = 50  # Number of training instances
        n_test = 60  # Number of test instances
        whichSet = 'SET2'  # Set identifier, must be set to 'SET2'
        setparam = 100  # Parameter related to "revenues"
        alphaE2 = 0.5  # Probability of building an edge

        ########### SMALL PARAM FOR TESTING ###########
        n=5
        n_test=5
        min_n_train=30
        max_n_train=40
        min_n_test = min_n_train
        max_n_test = max_n_train
        ########### SMALL PARAM FOR TESTING ###########

        if extend_training_instances:
            training_file_list = [f"data/{problem}/train", f"data/{problem}/more_train"]
        else:
            training_file_list = [f"data/{problem}/train"]

        for training_file in training_file_list:
            # Directory for training instances
            lp_dir_training = os.path.join(conf.ROOT_DIR, training_file)
            if os.path.exists(lp_dir_training):
                print(f"Cleaning directory: {lp_dir_training}")
                try:
                    shutil.rmtree(lp_dir_training)  # Recursively removes a directory and all its contents
                except Exception as e:
                    print(f"Error cleaning {lp_dir_training}: {e}")
            os.makedirs(lp_dir_training)

            # Generate training instances
            if problem == "gisp":
                build_gisp_instances.generate_instances(n, whichSet, setparam, alphaE2, min_n_train, max_n_train, er_prob, None, lp_dir_training, False) #GISP
            elif problem == "wpsm":
                build_wpsm_instances.generate_instances(n, min_n_train, max_n_train, lp_dir_training, False, er_prob) #WPMS
            elif problem == "fcmcnf":
                build_fcmcnf_instances.generate_instances(n, ntrain, ntrain, n_commodities_train, n_commodities_train, er_prob, lp_dir_training, True) #FCMCNF

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
        os.makedirs(lp_dir_test)

        # Generate test instances

        if problem == "gisp":
            build_gisp_instances.generate_instances(n_test, whichSet, setparam, alphaE2, min_n_test, max_n_test, er_prob, None, lp_dir_test, False) #GISP
        elif problem == "wpsm":
            build_wpsm_instances.generate_instances(n_test, min_n_test, max_n_test, lp_dir_test, False, er_prob) #WPMS
        elif problem == "fcmcnf":
            build_fcmcnf_instances.generate_instances(n, ntest, ntest, n_commodities_test, n_commodities_test, er_prob, lp_dir_test, True) #FCMCNF
