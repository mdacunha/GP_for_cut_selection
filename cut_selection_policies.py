from pyscipopt.scip import Cutsel
from pyscipopt import SCIP_RESULT
import random
import time
from operator import *
import re
import numpy as np
import os
import json
from filelock import FileLock
from num_cut_heuristic import num_cut_heuristic
import torch

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

class CustomCutSelector(Cutsel):

    def __init__(self, comp_policy, num_cuts_per_round=10, test=False, RL=False, nnet=None, inputs_type="",is_Test=False,
                 final_test=False, get_scores=False, heuristic=False, args={}, exp=0, parallel_filtering=False, 
                 min_orthogonality_root=0.9, min_orthogonality=0.9):
        super().__init__()
        self.comp_policy = comp_policy
        self.num_cuts_per_round = num_cuts_per_round
        self.min_orthogonality_root = min_orthogonality_root
        self.min_orthogonality = min_orthogonality
        self.test = test
        self.RL = RL
        self.nnet = nnet
        self.inputs_type = inputs_type
        self.is_Test = is_Test
        self.final_test = final_test
        self.get_scores = get_scores
        self.heuristic = heuristic
        self.end_time = 0
        self.args=args
        self.exp = exp
        self.parallel_filtering = parallel_filtering

        self.log_sample_list = []
        
        random.seed(42)
    
    def cutselselect(self, cuts, forcedcuts, root, maxnselectedcuts):
        """
        This is the main function used to select cuts. It must be named cutselselect and is called by default when
        SCIP performs cut selection if the associated cut selector has been included (assuming no cutsel with higher
        priority was called successfully before). This function aims to add self.num_cuts_per_round many cuts to
        the LP per round, prioritising the highest ranked cuts. It adds the highest ranked cuts, filtering by
        parallelism. In the case when not enough cuts are added and all the remaining cuts are too parallel,
        we simply add those with the highest score.
        @param cuts: These are the optional cuts we get to select from
        @type cuts: List of pyscipopt rows
        @param forcedcuts: These are the cuts that must be added
        @type forcedcuts: List of pyscipopt rows
        @param root: Boolean for whether we're at the root node or not
        @type root: Bool
        @param maxnselectedcuts: Maximum number of selected cuts
        @type maxnselectedcuts: int
        @return: Dictionary containing the keys 'cuts', 'nselectedcuts', result'. Warning: Cuts can only be reordered!
        @rtype: dict
        """
        if not self.parallel_filtering:
            # Initialise number of selected cuts and number of cuts that are still valid candidates
            n_cuts = len(cuts)
            nselectedcuts = 0

            # Generate the scores of each cut and thereby the maximum score
            # max_forced_score, forced_scores = self.scoring(forcedcuts)
            max_non_forced_score, scores = self.scoring(cuts, self.test)

            if root:
                c = 4
            else:
                c = 1

            if self.RL:
                if self.inputs_type == "only_scores":
                    inputs = np.array([score for score in scores])
                elif self.inputs_type == "only_features":
                    inputs = self.getfeatures(cuts)
                elif self.inputs_type == "scores_and_features":
                    inputs = self.get_scores_and_features(cuts, scores)
                else:
                    inputs = None

                #start_time = time.time()
                if self.exp==0:
                    if self.is_Test and self.final_test:
                        self.k = self.nnet.predict(inputs, mode="final_test")
                        num_cut = round(n_cuts * self.k)
                    elif self.is_Test and not self.final_test:
                        self.k = self.nnet.predict(inputs, mode="test")
                        num_cut = round(n_cuts * self.k)
                    else:
                        self.k = self.nnet.predict(inputs, mode="train")
                        self.log_sample_list.append(self.k)
                        try:
                            num_cut = int(torch.round(n_cuts * self.k))
                        except:
                            print("n_cuts", n_cuts, "k", self.k, flush=True)
                else:
                    if self.is_Test and self.final_test:
                        num_cut = self.nnet.predict(inputs, mode="final_test")
                        num_cut = int(num_cut/100 * n_cuts)
                    elif self.is_Test and not self.final_test:
                        num_cut = self.nnet.predict(inputs, mode="test")
                        num_cut = int(num_cut/100 * n_cuts)
                    else:
                        self.dist = self.nnet.predict(inputs, mode="train")
                        sample = self.dist.sample()
                        log_sample = self.dist.log_prob(sample)
                        self.log_sample_list.append(log_sample)
                        num_cut = sample.cpu().item() + 1
                        num_cut = int(num_cut/100 * n_cuts)
                #self.end_time += time.time() - start_time
            else:
                num_cut = c * self.num_cuts_per_round
                
            # Get the number of cuts that we will select this round.
            num_cuts_to_select = min(maxnselectedcuts, max(num_cut - len(forcedcuts), 0), n_cuts)
            #print(num_cuts_to_select)

            # Initialises parallel thresholds. Any cut with 'good' score can be at most good_max_parallel to a previous cut,
            # while normal cuts can be at most max_parallel. (max_parallel >= good_max_parallel)
            if root:
                max_parallel = 1 - self.min_orthogonality_root
                good_max_parallel = max(0.5, max_parallel)
            else:
                max_parallel = 1 - self.min_orthogonality
                good_max_parallel = max(0.5, max_parallel)      

            if self.get_scores:
                print("scores.json", num_cuts_to_select, scores)
                self.ajouter_donnee_json("scores.json", num_cuts_to_select, scores)

            if self.heuristic:
                num = num_cut_heuristic(scores)
                #num_cuts_to_select = min(num, num_cuts_to_select)
                num_cuts_to_select = num

            good_score = max_non_forced_score

            # This filters out all cuts in cuts who are parallel to a forcedcut.
            for forced_cut in forcedcuts:
                n_cuts, cuts, scores = self.filter_with_parallelism(n_cuts, nselectedcuts, forced_cut, cuts,
                                                                    scores, max_parallel, good_max_parallel, good_score)

            if maxnselectedcuts > 0 and num_cuts_to_select > 0:
                while n_cuts > 0:
                    # Break the loop if we have selected the required amount of cuts
                    if nselectedcuts == num_cuts_to_select:
                        break
                    # Re-sorts cuts and scores by putting the best cut at the beginning
                    cuts, scores = self.select_best_cut(n_cuts, nselectedcuts, cuts, scores)
                    nselectedcuts += 1
                    n_cuts -= 1
                    n_cuts, cuts, scores = self.filter_with_parallelism(n_cuts, nselectedcuts, cuts[nselectedcuts - 1],
                                                                        cuts,
                                                                        scores, max_parallel, good_max_parallel,
                                                                        good_score)

                # So far we have done the algorithm from the default method. We will now enforce choosing the highest
                # scored cuts from those that were previously removed for being too parallel.
                # Reset the n_cuts counter
                n_cuts = len(cuts) - nselectedcuts
                for remaining_cut_i in range(nselectedcuts, num_cuts_to_select):
                    cuts, scores = self.select_best_cut(n_cuts, nselectedcuts, cuts, scores)
                    nselectedcuts += 1
                    n_cuts -= 1
            
            return {'cuts': cuts, 'nselectedcuts': nselectedcuts,
                'result': SCIP_RESULT.SUCCESS}

        elif self.parallel_filtering:
            # Initialise number of selected cuts and number of cuts that are still valid candidates
            n_cuts = len(cuts)
            nselectedcuts = 0

            # Generate the scores of each cut and thereby the maximum score
            # max_forced_score, forced_scores = self.scoring(forcedcuts)
            max_non_forced_score, scores = self.scoring(cuts, self.test)

            good_score = max_non_forced_score

            if root:
                max_parallel = 1 - self.min_orthogonality_root
                good_max_parallel = max(0.5, max_parallel)
            else:
                max_parallel = 1 - self.min_orthogonality
                good_max_parallel = max(0.5, max_parallel)      

            if self.get_scores:
                print("scores.json", num_cuts_to_select, scores)
                self.ajouter_donnee_json("scores.json", num_cuts_to_select, scores)

            if self.heuristic:
                num = num_cut_heuristic(scores)
                #num_cuts_to_select = min(num, num_cuts_to_select)
                num_cuts_to_select = num

            # This filters out all cuts in cuts who are parallel to a forcedcut.
            for forced_cut in forcedcuts:
                n_cuts, cuts, scores = self.filter_with_parallelism(n_cuts, nselectedcuts, forced_cut, cuts,
                                                                    scores, max_parallel, good_max_parallel, good_score)

            if maxnselectedcuts > 0:
                while n_cuts > 0:
                    # Break the loop if we have selected the required amount of cuts
                    """if nselectedcuts == num_cuts_to_select:
                        break"""
                    if nselectedcuts >= maxnselectedcuts:
                        break
                    # Re-sorts cuts and scores by putting the best cut at the beginning
                    cuts, scores = self.select_best_cut(n_cuts, nselectedcuts, cuts, scores)
                    nselectedcuts += 1
                    n_cuts -= 1
                    n_cuts, cuts, scores = self.filter_with_parallelism(n_cuts, nselectedcuts, cuts[nselectedcuts - 1],
                                                                        cuts,
                                                                        scores, max_parallel, good_max_parallel,
                                                                        good_score)
                    
                # So far we have done the algorithm from the default method. We will now enforce choosing the highest
                # scored cuts from those that were previously removed for being too parallel.
                # Reset the n_cuts counter
                n_cuts = len(cuts) - nselectedcuts
                n_best_cuts_sorted = nselectedcuts
                for remaining_cut_i in range(nselectedcuts, len(cuts)):
                    cuts, scores = self.select_best_cut(n_cuts, nselectedcuts, cuts, scores)
                    nselectedcuts += 1
                    n_cuts -= 1

                n_cuts = len(cuts)

                if root:
                    c = 4
                else:
                    c = 1

                if self.RL:
                    if self.inputs_type == "only_scores":
                        L=[]
                        for (i, score) in enumerate(scores):
                            L.append([score, 1 if i < n_best_cuts_sorted else 0])
                        inputs = np.array(L)
                    elif self.inputs_type == "only_features":
                        inputs = self.getfeatures(cuts, nselectedcuts)
                    elif self.inputs_type == "scores_and_features":
                        inputs = self.get_scores_and_features(cuts, scores, nselectedcuts)
                    else:
                        inputs = None

                    #start_time = time.time()
                    if self.exp==0:
                        if self.is_Test and self.final_test:
                            self.k = self.nnet.predict(inputs, mode="final_test")
                            num_cut = round(n_cuts * self.k)
                        elif self.is_Test and not self.final_test:
                            self.k = self.nnet.predict(inputs, mode="test")
                            num_cut = round(n_cuts * self.k)
                        else:
                            self.k = self.nnet.predict(inputs, mode="train")
                            self.log_sample_list.append(self.k)
                            try:
                                num_cut = int(torch.round(n_cuts * self.k))
                            except:
                                print("n_cuts", n_cuts, "k", self.k, flush=True)
                    else:
                        if self.is_Test and self.final_test:
                            num_cut = self.nnet.predict(inputs, mode="final_test")
                            num_cut = int(num_cut/100 * n_cuts)
                        elif self.is_Test and not self.final_test:
                            num_cut = self.nnet.predict(inputs, mode="test")
                            num_cut = int(num_cut/100 * n_cuts)
                        else:
                            self.dist = self.nnet.predict(inputs, mode="train")
                            sample = self.dist.sample()
                            log_sample = self.dist.log_prob(sample)
                            self.log_sample_list.append(log_sample)
                            num_cut = sample.cpu().item() + 1
                            num_cut = int(num_cut/100 * n_cuts)
                    #self.end_time += time.time() - start_time
                else:
                    num_cut = c * self.num_cuts_per_round
                    
                # Get the number of cuts that we will select this round.
                num_cuts_to_select = min(maxnselectedcuts, max(num_cut - len(forcedcuts), 0), n_cuts)
                #print(num_cuts_to_select)

            return {'cuts': cuts, 'nselectedcuts': num_cuts_to_select,
                    'result': SCIP_RESULT.SUCCESS}
    
    def get_log_sample_list(self):
        return self.log_sample_list
    def time(self):
        return self.end_time

    def scoring(self, cuts, test=False):

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
                    
        scores = [0.0] * len(cuts)
        max_score = 0.0
           
        for i, cut in enumerate(cuts):      
            if not test:
                context = self.getcontext(cut, ind=1)
                score = evaluate_expression(self.comp_policy, context)
            else:
                efficacy_weight=1.0
                objparal_weight=0.1
                intsupport_weight=0.1
                dircutoffdist_weight=0.0

                context = self.getcontext(cut, ind=1)

                # Score pondéré
                score = (
                    efficacy_weight * context['getEfficacy'] + 
                    objparal_weight * context['getObjParallelism'] +
                    intsupport_weight * context['getNumIntCols'] / context['getNNonz'] +
                    dircutoffdist_weight * max(context['getEfficacy'], context['getCutLPSolCutoffDistance'])
                )
    
            score += 1e-4 if cut.isInGlobalCutpool() else 0
            score += random.uniform(0, 1e-6)
            max_score = max(max_score, score)
            scores[i] = score

            

        return max_score, scores
    
    def getfeatures(self, cuts, nselectedcuts=None):
        features = []
        if nselectedcuts is not None:
            for i, cut in enumerate(cuts):
                context = self.getcontext(cut, ind=2)
                features.append(np.array([1 if i < nselectedcuts else 0] + [context[key] for key in context.keys()]))
        else:
            for cut in cuts:
                context = self.getcontext(cut, ind=2)
                features.append(np.array([context[key] for key in context.keys()]))
        return features
    
    def get_scores_and_features(self, cuts, scores, nselectedcuts=None):
        features = []
        if nselectedcuts is not None:
            for i, cut in enumerate(cuts):
                context = self.getcontext(cut, ind=2)
                features.append(np.array([scores[i]] + [1 if i < nselectedcuts else 0] + [context[key] for key in context.keys()]))
        else:
            for i, cut in enumerate(cuts):
                context = self.getcontext(cut, ind=2)
                features.append(np.array([scores[i]] + [context[key] for key in context.keys()]))
        return features

    def getcontext(self, cut, ind=1):
        def compute_cut_violation(model, row):
            # Récupérer les variables et les coefficients (α)
            cols = row.getCols()  # Récupère les colonnes associées à la ligne
            vals = row.getVals()  # Récupère les coefficients correspondants

            # Extraire les variables des colonnes
            vars_in_cut = [col.getVar() for col in cols]

            # Calcul de αᵗ x* (solution LP)
            alphaTx = sum(model.getSolVal(None, var) * val for var, val in zip(vars_in_cut, vals))

            # Identifier le sens de l'inégalité et β
            lhs = row.getLhs()
            rhs = row.getRhs()

            if lhs == -model.infinity():
                # αᵗ x ≤ β
                beta = rhs
            elif rhs == model.infinity():
                # αᵗ x ≥ β
                beta = lhs
            else:
                # αᵗ x = β
                beta = lhs  # ou rhs

            # Calcul de la violation pondérée
            violation = (alphaTx - beta)/abs(beta) if beta != 0 else alphaTx
            score = max(0.0, violation)

            return score

        def get_cut_coeff_stats(row):

            coefs = row.getVals()

            # Si pas de coefficients trouvés
            if not coefs:
                return {'mean': 0.0, 'max': 0.0, 'min': 0.0, 'std': 0.0}

            stats = {
                'mean': np.mean(coefs),
                'max': np.max(coefs),
                'min': np.min(coefs),
                'std': np.std(coefs),
            }
            return stats
        
        def get_obj_coeff_stats(model):
            vars = model.getVars()
            coefs = [var.getObj() for var in vars]


            stats = {
                'mean': np.mean(coefs),
                'max': np.max(coefs),
                'min': np.min(coefs),
                'std': np.std(coefs),
            }
            return stats

        #getDualboundRoot = self.model.getDualboundRoot()
        
        sol = self.model.getBestSol() #if self.model.getNSols() > 0 else None
        if ind != 0:
            getNConss = self.model.getNConss()
            getNVars = self.model.getNVars()

        try:
            cutoffdist = self.model.getCutLPSolCutoffDistance(cut, sol)
        except:
            cutoffdist = 0.0  # pas toujours dispo
            print("Cutsel: Cut does not have a cutoff distance, using 0.0")      

        if ind == 0:
            context = {
                'getEfficacy': self.model.getCutEfficacy(cut),
                'getNumIntCols': self.model.getRowNumIntCols(cut),
                'getCutLPSolCutoffDistance': cutoffdist,
                'getObjParallelism': self.model.getRowObjParallelism(cut),
                'getCutViolation': compute_cut_violation(self.model, cut),
                "10000000":10000000
            }
        elif ind==1:
            context = {
                'getDepth': self.model.getDepth(),
                'getNConss': getNConss,
                'getNVars': getNVars,
                'getNNonz': cut.getNNonz(),
                'getEfficacy': self.model.getCutEfficacy(cut),
                'getNumIntCols': self.model.getRowNumIntCols(cut),
                'getCutLPSolCutoffDistance': cutoffdist,
                'getObjParallelism': self.model.getRowObjParallelism(cut),
                'getCutViolation': compute_cut_violation(self.model, cut),
                "10000000":10000000
            }

        elif ind==2:
            context = {
                    'getDepth': self.model.getDepth(),
                    'getNConss': getNConss,
                    'getNVars': getNVars,
                    'getNNonz': cut.getNNonz(),
                    'getEfficacy': self.model.getCutEfficacy(cut),
                    'getNumIntCols': self.model.getRowNumIntCols(cut),
                    'getCutLPSolCutoffDistance': cutoffdist,
                    'getObjParallelism': self.model.getRowObjParallelism(cut),
                    'getCutViolation': compute_cut_violation(self.model, cut),
                    'mean_cut_values' : get_cut_coeff_stats(cut)['mean'],
                    'max_cut_values' : get_cut_coeff_stats(cut)['max'],
                    'min_cut_values' : get_cut_coeff_stats(cut)['min'],
                    'std_cut_values' : get_cut_coeff_stats(cut)['std'],
                    'mean_obj_values' : get_obj_coeff_stats(self.model)['mean'],
                    'max_obj_values' : get_obj_coeff_stats(self.model)['max'],
                    'min_obj_values' : get_obj_coeff_stats(self.model)['min'],
                    'std_obj_values' : get_obj_coeff_stats(self.model)['std']
                    #"10000000":10000000
                }
        elif ind==3:
            context = {
                    'getEfficacy': self.model.getCutEfficacy(cut),
                    'getNumIntCols': self.model.getRowNumIntCols(cut),
                    'getCutLPSolCutoffDistance': cutoffdist,
                    'getObjParallelism': self.model.getRowObjParallelism(cut),
                    'getCutViolation': compute_cut_violation(self.model, cut),
                    'mean_cut_values' : get_cut_coeff_stats(cut)['mean'],
                    'max_cut_values' : get_cut_coeff_stats(cut)['max'],
                    'min_cut_values' : get_cut_coeff_stats(cut)['min'],
                    'std_cut_values' : get_cut_coeff_stats(cut)['std'],
                    'mean_obj_values' : get_obj_coeff_stats(self.model)['mean'],
                    'max_obj_values' : get_obj_coeff_stats(self.model)['max'],
                    'min_obj_values' : get_obj_coeff_stats(self.model)['min'],
                    'std_obj_values' : get_obj_coeff_stats(self.model)['std']
                }

        return context
    
        

    def filter_with_parallelism(self, n_cuts, nselectedcuts, cut, cuts, scores, max_parallel, good_max_parallel,
                                good_score):
        """
        Filters the given cut list by any cut_iter in cuts that is too parallel to cut. It does this by moving the
        parallel cut to the back of cuts, and decreasing the indices of the list that are scanned over.
        For the main portion of our selection we then never touch these cuts. In the case of us wanting to
        forcefully select an amount which is impossible under this filtering method however, we simply select the
        remaining highest scored cuts from the supposed untouched cuts.
        @param n_cuts: The number of cuts that are still viable candidates
        @type n_cuts: int
        @param nselectedcuts: The number of cuts already selected
        @type nselectedcuts: int
        @param cut: The cut which we will add, and are now using to filter the remaining cuts
        @type cut: pyscipopt row
        @param cuts: The list of cuts
        @type cuts: List of pyscipopt rows
        @param scores: The scores of each cut
        @type scores: List of floats
        @param max_parallel: The maximum allowed parallelism for non good cuts
        @type max_parallel: Float
        @param good_max_parallel: The maximum allowed parallelism for good cuts
        @type good_max_parallel: Float
        @param good_score: The benchmark of whether a cut is 'good' and should have it's allowed parallelism increased
        @type good_score: Float
        @return: The now number of viable cuts, the complete list of cuts, and the complete list of scores
        @rtype: int, list of pyscipopt rows, list of pyscipopt rows
        """
        # Go backwards through the still viable cuts.
        for i in range(nselectedcuts + n_cuts - 1, nselectedcuts - 1, -1):
            cut_parallel = self.model.getRowParallelism(cut, cuts[i])
            # The maximum allowed parallelism depends on the whether the cut is 'good'
            allowed_parallel = good_max_parallel if scores[i] >= good_score else max_parallel
            if cut_parallel > allowed_parallel:
                # Throw the cut to the end of the viable cuts and decrease the number of viable cuts
                cuts[nselectedcuts + n_cuts - 1], cuts[i] = cuts[i], cuts[nselectedcuts + n_cuts - 1]
                scores[nselectedcuts + n_cuts - 1], scores[i] = scores[i], scores[nselectedcuts + n_cuts - 1]
                n_cuts -= 1

        return n_cuts, cuts, scores

    def select_best_cut(self, n_cuts, nselectedcuts, cuts, scores):
        """
        Moves the cut with highest score which is still considered viable (not too parallel to previous cuts) to the
        front of the list. Note that 'front' here still has the requirement that all added cuts are still behind it.
        @param n_cuts: The number of still viable cuts
        @type n_cuts: int
        @param nselectedcuts: The number of cuts already selected to be added
        @type nselectedcuts: int
        @param cuts: The list of cuts themselves
        @type cuts: List of pyscipopt rows
        @param scores: The scores of each cut
        @type scores: List of floats
        @return: The re-sorted list of cuts, and the re-sorted list of scores
        @rtype: List of pyscipopt rows, list of floats
        """
        # Initialise the best index and score
        best_pos = nselectedcuts
        best_score = scores[nselectedcuts]
        for i in range(nselectedcuts + 1, nselectedcuts + n_cuts):
            if scores[i] > best_score:
                best_pos = i
                best_score = scores[i]
        # Move the cut with highest score to the front of the still viable cuts
        cuts[nselectedcuts], cuts[best_pos] = cuts[best_pos], cuts[nselectedcuts]
        scores[nselectedcuts], scores[best_pos] = scores[best_pos], scores[nselectedcuts]
        return cuts, scores
    
    def ajouter_donnee_json(self, fichier, cle, valeurs):
        valeurs_triees = sorted(valeurs, reverse=True)
        nouvelle_entree = {str(cle): valeurs_triees}

        # Fichier de verrouillage
        lock_path = fichier + ".lock"
        lock = FileLock(lock_path)

        with lock:  # Section critique protégée
            # Charger les données existantes si le fichier existe
            if os.path.exists(fichier):
                with open(fichier, "r") as f:
                    try:
                        data = json.load(f)
                        if not isinstance(data, list):
                            raise ValueError("Le fichier JSON doit contenir une liste.")
                    except json.JSONDecodeError:
                        data = []  # Fichier vide ou corrompu, on repart à zéro
            else:
                data = []

            # Ajouter la nouvelle entrée
            data.append(nouvelle_entree)

            # Écrire le tout dans le fichier
            with open(fichier, "w") as f:
                json.dump(data, f, indent=4)

"""class CustomCutSelector(Cutsel):
    
    def __init__(
        self,
        comp_policy,
        minortho=0.9,
        seed=42,
        num_cuts_per_round=None
    ):
        super().__init__()
        self.minortho = minortho
        self.comp_policy = comp_policy
        self.random = random.Random(seed)

    def copy(self):
        return CustomCutSelector(
            minortho=self.minortho,
            seed=42  # ou vous pouvez copier self.random.getstate()
        )

    def select(self, cuts):
        print("CustomCutSelector: select called with", len(cuts), "cuts")
        scored = []
        for cut in cuts:

            score = self.scoring(cut)

            # Biais si issue du pool global
            if cut.isInGlobalCutpool():
                score += 1e-4

            # Bruit aléatoire
            score += self.random.uniform(0.0, 1e-6)

            scored.append((cut, score))

        # Tri décroissant
        scored.sort(key=lambda x: x[1], reverse=True)

        # Sélection avec filtrage d’orthogonalité
        selected = []
        row_vectors = []  # vecteurs des coupes déjà sélectionnées

        for cut, score in scored:
            rowvec = cut.getDenseRepresentation()
            is_orthogonal = True
            for other in row_vectors:
                dot = sum(a * b for a, b in zip(rowvec, other))
                norm1 = math.sqrt(sum(a**2 for a in rowvec))
                norm2 = math.sqrt(sum(b**2 for b in other))
                if norm1 == 0 or norm2 == 0:
                    continue
                cos_theta = dot / (norm1 * norm2)
                if abs(cos_theta) > (1 - self.minortho):  # trop parallèle
                    is_orthogonal = False
                    break
            if is_orthogonal:
                selected.append(True)
                row_vectors.append(rowvec)
            else:
                selected.append(False)
            if self.num_cuts_per_round == sum(selected):
                break

        return selected
    
    def scoring(self, cut):

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
                
        # initialise the scoring of each cut as well as the max_score
        #scores = 0.0        
        
        #getDualboundRoot = self.model.getDualboundRoot()
        getNVars = self.model.getNVars()
        #sol = self.model.getBestSol() #if self.model.getNSols() > 0 else None
        getNConss = self.model.getNConss()

        try:
            cutoffdist = cut.getLPSolCutoffDistance()
        except:
            cutoffdist = 0.0  # pas toujours dispo

        # Cycle over all cuts and score them

        context = {
                'getDepth': self.model.getDepth(),
                'getNumIntCols': cut.getNumIntCols(),
                'getNConss': getNConss,
                'getNVars': getNVars,
                'getNNz': cut.getNNz(),
                'getEfficacy': cut.getEfficacy(),
                'getCutLPSolCutoffDistance': cutoffdist,
                'getObjParallelism': cut.getObjParallelism(),
                "10000000":10000000
            }
                
        score = evaluate_expression(self.comp_policy, context)

        score += 1e-4 if cut.isInGlobalCutpool() else 0
        score += random.uniform(0, 1e-6)

        return score

class test(Cutsel):
    def __init__(
        self,
        efficacy_weight=1.0,
        objparal_weight=0.1,
        intsupport_weight=0.1,
        dircutoffdist_weight=0.0,
        minortho=0.9,
        seed=42
    ):
        super().__init__()
        self.efficacy_weight = efficacy_weight
        self.objparal_weight = objparal_weight
        self.intsupport_weight = intsupport_weight
        self.dircutoffdist_weight = dircutoffdist_weight
        self.minortho = minortho
        self.random = random.Random(seed)

    def copy(self):
        return test(
            efficacy_weight=self.efficacy_weight,
            objparal_weight=self.objparal_weight,
            intsupport_weight=self.intsupport_weight,
            dircutoffdist_weight=self.dircutoffdist_weight,
            minortho=self.minortho,
            seed=42  # ou vous pouvez copier self.random.getstate()
        )

    def select(self, cuts):
        scored = []
        for cut in cuts:
            # Efficacité (distance / norme)
            efficacy = cut.getEfficacy()

            # Parallélisme avec l'objectif
            objparal = cut.getObjParallelism()

            # Support entier
            try:
                n_intcols = cut.getNumIntCols()
                nnz = cut.getNNZ()
                intsupport = n_intcols / nnz if nnz > 0 else 0.0
            except:
                intsupport = 0.0  # fallback si info non dispo

            # Distance coupure (optionnelle)
            try:
                cutoffdist = cut.getLPSolCutoffDistance()
            except:
                cutoffdist = 0.0  # pas toujours dispo

            # Score pondéré
            score = (
                self.efficacy_weight * efficacy +
                self.objparal_weight * objparal +
                self.intsupport_weight * intsupport +
                self.dircutoffdist_weight * max(efficacy, cutoffdist)
            )

            # Biais si issue du pool global
            if cut.isInGlobalCutpool():
                score += 1e-4

            # Bruit aléatoire
            score += self.random.uniform(0.0, 1e-6)

            scored.append((cut, score))

        # Tri décroissant
        scored.sort(key=lambda x: x[1], reverse=True)

        # Sélection avec filtrage d’orthogonalité
        selected = []
        row_vectors = []  # vecteurs des coupes déjà sélectionnées

        for cut, score in scored:
            rowvec = cut.getDenseRepresentation()
            is_orthogonal = True
            for other in row_vectors:
                dot = sum(a * b for a, b in zip(rowvec, other))
                norm1 = math.sqrt(sum(a**2 for a in rowvec))
                norm2 = math.sqrt(sum(b**2 for b in other))
                if norm1 == 0 or norm2 == 0:
                    continue
                cos_theta = dot / (norm1 * norm2)
                if abs(cos_theta) > (1 - self.minortho):  # trop parallèle
                    is_orthogonal = False
                    break
            if is_orthogonal:
                selected.append(True)
                row_vectors.append(rowvec)
            else:
                selected.append(False)

        return selected"""