import random
from typing import List, Optional, Set
from Operations import *

class Instruction:
    def __init__(self):
        self.root = self.create_random_tree()
        self.depth = self.count_depth()
        self.num_node = self.count_node()
        self.name = ""
        self.get_name()

    def create_random_tree(self, depth=1):
        ret_op = self.create_random_node(depth)
        for i in range(ret_op.num_children):
            ret_op.children[i] = self.create_random_tree(depth + 1)
        return ret_op

    def create_random_node(self, depth):
        available = [0, 1, 2, 4, 5, 6, 7]
        if depth == 1:
            return WriteReg()
        else:
            num_ftn_choices = len(available) if depth < MaxDepth else NUM_TERM
            randn = random.choice(available[:num_ftn_choices])

            return {
                0: ConstNode,
                1: InputNode,
                2: ReadReg,
                4: Add,
                5: Subtract,
                6: Multiply,
                7: Divide,
            }.get(randn, lambda: print("invalid random number, bad news"))()

    def get_name(self, node=None):
        if node is None:
            node = self.root
            self.name = ""

        if node.num_children == 2:
            self.name += "("
            self.get_name(node.children[0])
            self.name += node.get_label()
            self.get_name(node.children[1])
            self.name += ")"
        elif node.num_children == 1:
            self.name += node.get_label() + "("
            self.get_name(node.children[0])
            self.name += ")"
        elif node.num_children == 0:
            self.name += node.get_label()

    def print_instruction(self, node=None):
        if node is None:
            node = self.root

        if node.num_children == 2:
            print("(", end="")
            self.print_instruction(node.children[0])
            print(node.get_label(), end="")
            self.print_instruction(node.children[1])
            print(")", end="")
        elif node.num_children == 1:
            print(node.get_label() + "(", end="")
            self.print_instruction(node.children[0])
            print(")", end="")
        elif node.num_children == 0:
            print(node.get_label(), end="")
        if node == self.root:
            print()

    def delete_instruction(self, node=None):
        if node is None:
            node = self.root
        for child in node.children:
            self.delete_instruction(child)
        del node

    def dfs_visit(self, tsi, ci, node=None):
        if tsi > self.num_node - 1:
            print("the tsi is larger than the node number")
            return None
        if node is None:
            node = self.root
        if ci[0] == tsi:
            return node
        else:
            for i in range(node.num_children):
                ci[0] += 1
                result = self.dfs_visit(tsi, ci, node.children[i])
                if result is not None:
                    return result
        return None

    def get_subtree(self, si):
        cnt = [0]
        sub_root = self.dfs_visit(si, cnt)
        if cnt[0] < si:
            print(f"cnt {cnt[0]} <si {si}, numNode: {self.num_node}")
            print(f"root: {self.root.get_label()}")
        return sub_root.clone()

    def get_random_subtree(self):
        si = random.randint(0, self.num_node - 1)
        return self.get_subtree(si)

    def count_node(self, node=None):
        if node is None:
            node = self.root
        if node.num_children == 0:
            return 1
        return 1 + sum(self.count_node(child) for child in node.children)

    def count_depth(self, node=None, max_depth=0, depth=1):
        if node is None:
            node = self.root
        if depth > max_depth:
            max_depth = depth
        for child in node.children:
            max_depth = self.count_depth(child, max_depth, depth + 1)
        return max_depth

    def get_op_depth(self, target_op, depth=1, node=None):
        if node is None:
            node = self.root
        if target_op == node:
            return depth
        for child in node.children:
            tmp = self.get_op_depth(target_op, depth + 1, child)
            if tmp > 0:
                return tmp
        return -1

    def get_depth_set(self, ds: List[Op], tar_depth=1, node=None):
        if node is None:
            node = self.root
        if self.count_depth(node) <= tar_depth:
            ds.append(node)
        for child in node.children:
            self.get_depth_set(ds, tar_depth, child)

    def eval(self, ind=0):
        return self.root.eval(ind)

    def mutate_instruction(self, node=None, depth=1, keyop=None):
        if node is None:
            cnt = [0]
            keyop = self.dfs_visit(random.randint(0, self.num_node - 1), cnt)
            node = self.root

        if random.random() < MUTATION_THRESH or node == keyop:
            new_node = self.create_random_node(depth)
            children_to_move = min(new_node.num_children, node.num_children)

            for i in range(children_to_move):
                new_node.children[i] = node.children[i]
            for i in range(children_to_move, new_node.num_children):
                new_node.children[i] = self.create_random_tree(depth + 1)
            for i in range(children_to_move, node.num_children):
                self.delete_instruction(node.children[i])

            if node == keyop:
                keyop = None
            del node
            node = new_node
            self.root = node
            self.depth = self.count_depth()
            self.num_node = self.count_node()
            self.get_name()

        for i in range(node.num_children):
            self.mutate_instruction(node.children[i], depth + 1, keyop)

    def crossover(self, donating_instr, node=None, keyop=None):
        if node is None:
            cnt = [0]
            keyop = self.dfs_visit(1 + random.randint(0, self.num_node - 2), cnt)
            node = self.root.children[0]

        if random.random() < CROSSOVER_THRESH or node == keyop:
            ds = []
            donating_instr.get_depth_set(ds, MaxDepth - self.get_op_depth(node) + 1)
            si = random.choice(ds)
            new_subtree = si.clone()

            if node == keyop:
                keyop = None
            self.delete_instruction(node)
            node = new_subtree
            self.root = node
            self.depth = self.count_depth()
            self.num_node = self.count_node()
            self.get_name()
        else:
            for i in range(node.num_children):
                self.crossover(donating_instr, node.children[i], keyop)

    def de_mutate(self, tar_in_tree, r2_in_tree, node=None, depth=1, keyop=None):
        if node is None:
            cnt = [0]
            keyop = self.dfs_visit(random.randint(0, self.num_node - 1), cnt)
            node = self.root

        F = randval()
        CR = randval()
        dd1 = dd2 = 0
        if tar_in_tree and tar_in_tree.label != node.label:
            dd1 = 1
        if r2_in_tree and r2_in_tree.label != node.label:
            dd2 = 1
        c_vector = F * dd1 + F * dd2 - F * dd1 * F * dd2

        if randval() < CR or node == keyop:
            if randval() < c_vector or randval() < 0.03:
                new_node = self.create_random_node(depth)
                children_to_move = min(new_node.num_children, node.num_children)
                for i in range(children_to_move):
                    new_node.children[i] = node.children[i]
                for i in range(children_to_move, new_node.num_children):
                    new_node.children[i] = self.create_random_tree(depth + 1)
                for i in range(children_to_move, node.num_children):
                    self.delete_instruction(node.children[i])
                if node == keyop:
                    keyop = None
                del node
                node = new_node
                self.root = node
                self.depth = self.count_depth()
                self.num_node = self.count_node()
                self.get_name()

        for i in range(node.num_children):
            t = tar_in_tree.children[i] if tar_in_tree and i < tar_in_tree.num_children else None
            r = r2_in_tree.children[i] if r2_in_tree and i < r2_in_tree.num_children else None
            self.de_mutate(t, r, node.children[i], depth + 1, keyop)

    def __eq__(self, other):
        return self.name == other.name

    def __copy_from__(self, other):
        if other.root == self.root:
            return self
        if self.root:
            self.delete_instruction(self.root)
        self.depth = other.depth
        self.num_node = other.num_node
        self.name = other.name
        self.root = other.root.clone()
        return self

    def get_read_reg(self, node=None) -> Set[int]:
        if node is None:
            node = self.root
        res = set()
        if isinstance(node, ReadReg):
            res.add(node.reg_index)
        for child in node.children:
            res |= self.get_read_reg(child)
        return res

    def get_write_reg(self, node=None) -> Set[int]:
        if node is None:
            node = self.root
        res = set()
        if isinstance(node, WriteReg):
            res.add(node.reg_index)
        for child in node.children:
            res |= self.get_write_reg(child)
        return res
