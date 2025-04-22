import random

from pyscipopt import Nodesel
#import pyscipopt.scip as sp
import numpy as np
import time
#import torch
#from joblib import load
from operator import *
import re
import copy

#from learning_to_comparenodes.learning.model import GNNPolicy, RankNet

def protectedDiv(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1

operations = {
    'add': add,
    'mul': mul,
    'sub': sub,
    'protectedDiv': protectedDiv
}

class CustomNodeSelector(Nodesel):

    def __init__(self, sel_policy='', comp_policy=''):
        self.sel_policy = sel_policy
        self.comp_policy = comp_policy
        self.getDualboundRoot = 0
        self.getNConss = 0
        self.getNVars = 0
        self.started = False


    def nodeselect(self):#### Only BFS policy
        #print(self.model.getOpenNodes())
        if self.sel_policy == "BFS":
            return {'selnode': self.model.getBestNode()}
        else:
            open_nodes = self.model.getOpenNodes()
            if len(open_nodes[1]) !=0:
                return {'selnode': self.model.getBestChild()}
            if len(open_nodes[2]) !=0:
                return {'selnode': self.model.getBestSibling()}
            return {'selnode': self.model.getBestNode()}


    def nodecomp(self, node1, node2):
        if self.comp_policy == 'estimate':
            val1 = node1.getEstimate()
            val2 = node2.getEstimate()

        elif self.comp_policy == 'LB':
            val1 = node1.getLowerbound()
            val2 = node2.getLowerbound()
        else:
            if self.started is False:
              self.getDualboundRoot = self.model.getDualboundRoot()#copy.deepcopy(
              self.getNConss = self.model.getNConss()
              self.getNVars = self.model.getNVars()
              self.started = True
            val1 = self.tuned_policy(node1)
            val2 = self.tuned_policy(node2)


        if self.model.isLT(val1,val2):
            return -1
        if self.model.isGT(val1, val2):
            return 1
        elif (self.model.isInfinity(val1) and self.model.isInfinity(val2)) or \
            (self.model.isInfinity(-val1) and self.model.isInfinity(-val2)) or \
            self.model.isEQ(val1, val2):
            ntype1 = node1.getType()
            ntype2 = node2.getType()
            CHILD, SIBLING = 3, 2
            if (ntype1 == CHILD and ntype2 != CHILD) or (ntype1 == SIBLING and ntype2 != SIBLING):
                return -1
            elif (ntype1 != CHILD and ntype2 == CHILD) or (ntype1 != SIBLING and ntype2 == SIBLING):
                return 1
            else:
                return 0
        else:
            return 0

        #print(new_funct(100000,2,5,-2111,2342,98987,7658))

    def tuned_policy(self,node):

      return comp_policy_as_a_function(self.comp_policy,node.getDepth(),node.getEstimate(),node.getLowerbound(),self.getDualboundRoot,self.getNConss,self.getNVars)



def comp_policy_as_a_function(comp_policy,getDepth, getEstimate, getLowerbound, getDualboundRoot, getNConss, getNVars):

        delimiters = ["(", ",", " ", ")"]
        #string = self.comp_policy
        context = {
                'getDepth': getDepth,
                'getEstimate': getEstimate,
                'getLowerbound': getLowerbound,
                'getDualboundRoot': getDualboundRoot,
                'getNConss': getNConss,
                'getNVars': getNVars,
                "10000000":10000000
            }
        return evaluate_expression(comp_policy, context)


def parse_args(args, context):
    # Split arguments considering nested functions
    args_list = []
    nested = 0
    last_split = 0
    for i, char in enumerate(args):
        if char == '(':
            nested += 1
        elif char == ')':
            nested -= 1
        elif char == ',' and nested == 0:
            args_list.append(args[last_split:i].strip())
            last_split = i + 1
    args_list.append(args[last_split:].strip())
    return [evaluate_expression(arg, context) for arg in args_list]

def evaluate_expression(expr, context):
    expr = expr.strip()
    # Regex to match outermost function calls
    func_pattern = r'^(\w+)\((.*)\)$'
    match = re.match(func_pattern, expr)
    if match:
        func_name, args_str = match.groups()
        if func_name in operations:
            args = parse_args(args_str, context)
            return operations[func_name](*args)
        else:
            raise ValueError(f"Unsupported function '{func_name}'")
    else:
        try:
            return float(expr)
        except ValueError:
            return context[expr]


