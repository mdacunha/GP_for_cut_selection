import math
import random
import copy

# Constants (à adapter si besoin)
MaxChild = 2
MaxReg = 10
Inf = 1e6
in_dimen = 5  # Exemple de dimension des entrées

# Variables globales simulées
inputs = [0.0 for _ in range(in_dimen)]
registers = [0.0 for _ in range(MaxReg)]


class Op:
    def __init__(self):
        self.label = ""
        self.num_children = 0
        self.children = [None for _ in range(MaxChild)]

    def eval(self, index=0):
        raise NotImplementedError

    def clone(self):
        raise NotImplementedError

    def get_label(self):
        raise NotImplementedError


class ConstNode(Op):
    def __init__(self, pre_set_val=None):
        super().__init__()
        self.num_children = 0
        self.const_val = pre_set_val if pre_set_val is not None else random.random()
        self.label = f"{self.const_val:.6f}"

    def eval(self, index=0):
        return self.const_val

    def clone(self):
        return ConstNode(self.const_val)

    def get_label(self):
        return self.label


class InputNode(Op):
    def __init__(self, input_index=None):
        super().__init__()
        self.num_children = 0
        self.input_index = input_index if input_index is not None else random.randint(0, in_dimen - 1)
        self.set_values(self.input_index)

    def set_values(self, input_index):
        self.input_index = input_index
        self.label = f"x{input_index}"

    def eval(self, index=0):
        return inputs[self.input_index]

    def clone(self):
        return InputNode(self.input_index)

    def get_label(self):
        return self.label


class ReadReg(Op):
    def __init__(self, reg_index=None):
        super().__init__()
        self.num_children = 0
        self.reg_index = reg_index if reg_index is not None else random.randint(0, MaxReg - 1)
        self.set_values(self.reg_index)

    def set_values(self, reg_index):
        self.reg_index = reg_index
        self.label = f"v{reg_index}"

    def eval(self, index=0):
        res = registers[self.reg_index]
        if abs(res) > Inf:
            res = registers[self.reg_index] = Inf
        return res

    def clone(self):
        return ReadReg(self.reg_index)

    def get_label(self):
        return self.label


class WriteReg(Op):
    def __init__(self, reg_index=None):
        super().__init__()
        self.num_children = 1
        self.reg_index = reg_index if reg_index is not None else random.randint(0, MaxReg - 1)
        self.set_values(self.reg_index)

    def set_values(self, reg_index):
        self.reg_index = reg_index
        self.label = f"v{reg_index}="

    def eval(self, index=0):
        if self.children[0] is not None:
            res = self.children[0].eval(index)
            if abs(res) > Inf:
                res = Inf
            registers[self.reg_index] = res
            return res
        else:
            raise ValueError("Child is not defined in WriteReg")

    def clone(self):
        cloned = WriteReg(self.reg_index)
        cloned.children[0] = self.children[0].clone()
        return cloned

    def get_label(self):
        return self.label


class Add(Op):
    def __init__(self):
        super().__init__()
        self.num_children = 2
        self.label = "+"

    def eval(self, index=0):
        if self.children[0] and self.children[1]:
            a = self.children[0].eval(index)
            b = self.children[1].eval(index)
            res = a + b
            return res if abs(res) <= Inf else Inf
        else:
            raise ValueError("Children not defined in Add")

    def clone(self):
        cloned = Add()
        cloned.children = [child.clone() for child in self.children]
        return cloned

    def get_label(self):
        return self.label


class Subtract(Op):
    def __init__(self):
        super().__init__()
        self.num_children = 2
        self.label = "-"

    def eval(self, index=0):
        if self.children[0] and self.children[1]:
            a = self.children[0].eval(index)
            b = self.children[1].eval(index)
            res = a - b
            return res if abs(res) <= Inf else Inf
        else:
            raise ValueError("Children not defined in Subtract")

    def clone(self):
        cloned = Subtract()
        cloned.children = [child.clone() for child in self.children]
        return cloned

    def get_label(self):
        return self.label


class Multiply(Op):
    def __init__(self):
        super().__init__()
        self.num_children = 2
        self.label = "*"

    def eval(self, index=0):
        if self.children[0] and self.children[1]:
            a = self.children[0].eval(index)
            b = self.children[1].eval(index)
            res = a * b
            return res if abs(res) <= Inf else Inf
        else:
            raise ValueError("Children not defined in Multiply")

    def clone(self):
        cloned = Multiply()
        cloned.children = [child.clone() for child in self.children]
        return cloned

    def get_label(self):
        return self.label


class Divide(Op):
    def __init__(self):
        super().__init__()
        self.num_children = 2
        self.label = "/"

    def eval(self, index=0):
        if self.children[0] and self.children[1]:
            v0 = self.children[0].eval(index)
            v1 = self.children[1].eval(index)
            res = v0 / math.sqrt(1 + v1 ** 2)
            return res if abs(res) <= Inf else Inf
        else:
            raise ValueError("Children not defined in Divide")

    def clone(self):
        cloned = Divide()
        cloned.children = [child.clone() for child in self.children]
        return cloned

    def get_label(self):
        return self.label


class Sin(Op):
    def __init__(self):
        super().__init__()
        self.num_children = 1
        self.label = "sin"

    def eval(self, index=0):
        if self.children[0]:
            v = self.children[0].eval(index)
            res = math.sin(v)
            return res if abs(res) <= Inf else Inf
        else:
            raise ValueError("Child not defined in Sin")

    def clone(self):
        cloned = Sin()
        cloned.children[0] = self.children[0].clone()
        return cloned

    def get_label(self):
        return self.label


class Cos(Op):
    def __init__(self):
        super().__init__()
        self.num_children = 1
        self.label = "cos"

    def eval(self, index=0):
        if self.children[0]:
            v = self.children[0].eval(index)
            res = math.cos(v)
            return res if abs(res) <= Inf else Inf
        else:
            raise ValueError("Child not defined in Cos")

    def clone(self):
        cloned = Cos()
        cloned.children[0] = self.children[0].clone()
        return cloned

    def get_label(self):
        return self.label


class Exp(Op):
    def __init__(self):
        super().__init__()
        self.num_children = 1
        self.label = "exp"

    def eval(self, index=0):
        if self.children[0]:
            v = self.children[0].eval(index)
            v = min(v, 10)
            res = math.exp(v)
            return res if abs(res) <= Inf else Inf
        else:
            raise ValueError("Child not defined in Exp")

    def clone(self):
        cloned = Exp()
        cloned.children[0] = self.children[0].clone()
        return cloned

    def get_label(self):
        return self.label


class Ln(Op):
    def __init__(self):
        super().__init__()
        self.num_children = 1
        self.label = "ln"

    def eval(self, index=0):
        if self.children[0]:
            v = abs(self.children[0].eval(index))
            v = max(v, 1e-3)
            res = math.log(v)
            return res if abs(res) <= Inf else Inf
        else:
            raise ValueError("Child not defined in Ln")

    def clone(self):
        cloned = Ln()
        cloned.children[0] = self.children[0].clone()
        return cloned

    def get_label(self):
        return self.label
