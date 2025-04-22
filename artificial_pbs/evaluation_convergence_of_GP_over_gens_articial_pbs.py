import os
import json
import matplotlib.pyplot as plt
import conf

def convergence_of_gp_over_generations(simulation_folder,saving=True):
    folder = os.path.join(simulation_folder, "GP_function")
    file = os.listdir(folder)[0]
    path = folder+"\\"+file
    print(path)
    with open(path
            ,
              'r') as openfile:
        perfs = json.load(openfile)

    averages = []
    for elt in perfs[0]:
        averages.append(elt["min"])

    plt.plot(averages)
    plt.xlabel("nb of generation")
    plt.ylabel("Best SS policy fitness")
    plt.yscale("log")
    #plt.title(f"Best performance in the GP_function population through generations, for problem {problem}")
    if saving:
        saving_path = os.path.join(conf.ROOT_DIR,
                                     f'simulation_outcomes/{problem}/convergence_plot_{problem}.pdf')
        plt.savefig(saving_path, format="pdf", bbox_inches="tight")
    plt.show()
    print("the best GP function is ", perfs[1][0])
    return perfs[1][0]

if __name__ == "__main__":

    for problem in ["wpms","gisp","fcmcnf"]:
        simulation_folder = os.path.join(conf.ROOT_DIR, f"simulation_outcomes/{problem}")
        convergence_of_gp_over_generations(simulation_folder)