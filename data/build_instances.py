import argparse
import os

import data.build_gisp_instances, data.build_fcmcnf_instances, data.build_wpms_instances

def build_new_set_of_instances(problem,partition,nb_of_instances=50):
    n= nb_of_instances
    if problem == "gisp":
        exp_dir = f"../data/gisp/"
        whichSet = 'SET2'
        setparam = 100.0
        alphaE2 = 0.5
        er_prob = 0.6
        lp_dir = os.path.join(os.path.dirname(__file__), exp_dir + partition + '/')
        if not os.path.exists(lp_dir):
            os.makedirs(lp_dir)
        if partition == "test":
            min_n = 60
            max_n = 70
        elif partition == "transfer":
            min_n = 80
            max_n = 100
        else:
            print("ERROR, partition: ", partition)
        data.build_gisp_instances.generate_instances(n, whichSet, setparam, alphaE2, min_n, max_n, er_prob, None, lp_dir, False)
    elif problem == "fcmcnf":
        exp_dir = f"../data/fcmcnf/"
        er_prob = 0.3
        if partition == "test":
            min_n = 15
            max_n = 15
        elif partition == "transfer":
            min_n = 20
            max_n = 20
        else:
            print("ERROR, partition: ",partition)
        lp_dir = os.path.join(os.path.dirname(__file__), exp_dir + partition + '/')
        if not os.path.exists(lp_dir):
            os.makedirs(lp_dir)
        min_n_commodities = int(1.5 * max_n)
        max_n_commodities = int(1.5 * max_n)
        data.build_fcmcnf_instances.generate_instances(n, min_n, max_n, min_n_commodities, max_n_commodities, er_prob,
                           lp_dir, True)
    elif problem == "wpms":
        exp_dir = f"../data/wpms/"
        er_prob = 0.6
        if partition == "test":
            min_n = 60
            max_n = 70

        elif partition == "transfer":
            min_n = 70
            max_n = 80
        else:
            print("ERROR, partition: ",partition)
        lp_dir = os.path.join(os.path.dirname(__file__), exp_dir + partition + '/')
        if not os.path.exists(lp_dir):
            os.makedirs(lp_dir)
        data.build_wpms_instances.generate_instances(n, min_n, max_n, lp_dir, False, er_prob)
    else:
        print("ERROR, problem: ",problem)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="generate new instances.")
    parser.add_argument('problem', type=str, help='problem')
    parser.add_argument('partition', type=str, help='partition')

    args = parser.parse_args()
    build_new_set_of_instances(args.problem, args.partition)