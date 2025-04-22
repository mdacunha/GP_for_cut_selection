import os
import sys
import time

import networkx as nx
import random
import pyscipopt as sp

import math

def dimacsToNx(filename):
    g = nx.Graph()
    with open(filename, 'r') as f:
        for line in f:
            arr = line.split()
            if line[0] == 'e':
                g.add_edge(int(arr[1]), int(arr[2]))
    return g


def generateRevsCosts(g, whichSet, setParam):
    if whichSet == 'SET1':
        for node in g.nodes():
            g.nodes[node]['revenue'] = random.randint(1, 100)
        for u, v, edge in g.edges(data=True):
            edge['cost'] = (g.node[u]['revenue'] /
                            + g.node[v]['revenue'])/float(setParam)
    elif whichSet == 'SET2':
        for node in g.nodes():
            g.nodes[node]['revenue'] = float(setParam)
        for u, v, edge in g.edges(data=True):
            edge['cost'] = 1.0


def generateE2(g, alphaE2):
    E2 = set()
    for edge in g.edges():
        if random.random() <= alphaE2:
            E2.add(edge)
    return E2


def createIP(g, E2, ipfilename):
    #print(ipfilename)
    with open(ipfilename, 'w') as lp_file:
        val = 100
        lp_file.write("maximize\nOBJ:")
        lp_file.write("100x0")
        count = 0
        for node in g.nodes():
            if count:
                lp_file.write(" + " + str(val) + "x" + str(node))
            count += 1
        for edge in E2:
            lp_file.write(" - y" + str(edge[0]) + '_' + str(edge[1]))
        lp_file.write("\n Subject to\n")
        constraint_count = 1
        for node1, node2, edge in g.edges(data=True):
            if (node1, node2) in E2:
                lp_file.write("C" + str(constraint_count) + ": x" + str(node1)
                              + "+x" + str(node2) + "-y" + str(node1) + "_"
                              + str(node2) + " <=1 \n")
            else:
                lp_file.write("C" + str(constraint_count) + ": x" + str(node1)
                              + "+" + "x" + str(node2) + " <=1 \n")
            constraint_count += 1

        lp_file.write("\nbinary\n")
        for node in g.nodes():
            lp_file.write(f"x{node}\n")
            
def generate_instances(nb_of_instances, whichSet, setParam, alphaE2, min_n, max_n, er_prob, instance, lp_dir, solve) :
    #initial_time =time.time()
    for i in range(nb_of_instances):
        solved = False
        while solved is False:
            solved = True
            seed = math.floor(random.random() * 2 ** 31)
            if instance is None:
                # Generate random graph
                numnodes = random.randint(min_n, max_n)
                g = nx.erdos_renyi_graph(n=numnodes, p=er_prob)
                lpname = ("er_n=%d_m=%d_p=%.2f_%s_setparam=%.2f_alpha=%.2f"
                        % (numnodes, nx.number_of_edges(g), er_prob, whichSet,
                            setParam, alphaE2))
            else:
                g = dimacsToNx(instance)
                # instanceName = os.path.splitext(instance)[1]
                instanceName = instance.split('/')[-1]
                lpname = ("%s_%s_%g_%g" % (instanceName, whichSet, alphaE2,
                        setParam))
            #print(lpname)

            # Generate node revenues and edge costs
            generateRevsCosts(g, whichSet, setParam)
            # Generate the set of removable edges
            E2 = generateE2(g, alphaE2)
            # Create IP, write it to file, and solve it with CPLEX
            #print(lpname)
            # ip = createIP(g, E2, lp_dir + "/" + lpname)
            createIP(g, E2, lp_dir + "\\" + lpname + ".lp")
            if solve:
                model = sp.Model()
                model.hideOutput()
                model.readProblem(lp_dir +"\\" + lpname + ".lp")
                model.optimize()
                model.writeBestSol(lp_dir +"\\" + lpname + ".sol")

                if model.getNNodes() <= 1:
                    os.remove(lp_dir +"\\" + lpname + ".lp")
                    os.remove(lp_dir +"\\" + lpname  + ".sol")
                    solved = False


if __name__ == "__main__":
    exp_dir = f"..\\data\\gisp\\"

    whichSet = 'SET2'
    setparam = 100.0
    alphaE2 = 0.5
    n = 50


    er_prob = 0.6

    data_partition = "train"
    min_n = 200
    max_n = 250
    lp_dir = os.path.join(os.path.dirname(__file__), exp_dir + data_partition + '\\')
    generate_instances(n, whichSet, setparam, alphaE2, min_n, max_n, er_prob, None, lp_dir, False)


        

