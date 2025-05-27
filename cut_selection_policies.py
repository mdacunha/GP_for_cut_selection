from pyscipopt.scip import Cutsel
import random
import math
from operator import *
import re


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

        return selected
