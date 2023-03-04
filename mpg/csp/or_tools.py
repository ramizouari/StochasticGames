import itertools
from typing import List, Any, Tuple

from ortools.sat.python import cp_model
from csp import max_atom, Variable
import games.mpg as mpg
import numpy as np


class MaxAtomCpModel:

    def __init__(self):
        self.formal_variables = set()
        self.constraints: List[Tuple[Variable, List[Variable], Any]] = []
        self.radius = 0

    def addMaxAtom(self, x: Variable, Y: List[Variable], c: Any):
        self.constraints.append((x, Y, c))
        self.formal_variables.add(x)
        self.formal_variables.update(Y)
        self.radius += abs(c)

    def solve(self):

        dnf = [Y for x, Y, c in self.constraints]
        print(f"Trying {np.prod([len(Y) for Y in dnf])} combinations")
        for k, Y in enumerate(itertools.product(*dnf)):
            if k % 1000 == 0:
                print(f"Trying {k}th combination")
            model = cp_model.CpModel()
            variables = {v: model.NewIntVar(-self.radius, self.radius, v.name) for v in self.formal_variables}
            for C, y in zip(self.constraints, Y):
                x, _, c = C
                model.Add(variables[x] <= variables[y] + c)
            solver = cp_model.CpSolver()
            status = solver.Solve(model)
            if status == cp_model.OPTIMAL:
                return solver, status, variables
        raise Exception("No solution found")


def powerset(iterable):
    s = set(iterable)
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s) + 1))

class MinMaxCpModel:

    def __init__(self):
        self.formal_variables = set()
        self.constraints: List[Tuple[Variable, List[Variable], Any]] = []
        self.radius = 0

    def addMaxAtom(self, x: Variable, Y: List[Variable], C: List[Any]):
        self.constraints.append((x, Y, C))
        self.formal_variables.add(x)
        self.formal_variables.update(Y)
        self.radius += np.sum(np.abs(C))

    def addMinAtom(self, x: Variable, Y: List[Variable], C: List[Any]):
        for y, c in zip(Y, C):
            self.constraints.append((x, [y], [c]))
        self.formal_variables.add(x)
        self.formal_variables.update(Y)
        self.radius += np.sum(np.abs(C))

    def solve(self):

        dnf = [[(y,c) for y,c in zip(Y,C)]for x, Y, C in self.constraints]
        # Iterate over all possible combinations of variables
        k=0
        print(f"Trying {2**len(self.formal_variables)*np.prod([len(C) for C in dnf])} combinations")
        for V in powerset(self.formal_variables):
            # Variables in V are fixed to -inf
            # Iterate over all possible combinations of constraints
            for Z in itertools.product(*dnf):
                k+=1
                if k % 10000 == 0:
                    print(f"Trying {k}th combination")
                model = cp_model.CpModel()
                variables = {v: model.NewIntVar(-self.radius, self.radius, v.name) for v in self.formal_variables}
                potentially_feasible=True
                for R, z in zip(self.constraints, Z):
                    y, c = z
                    x, _, _ = R
                    if y in V and x not in V:
                        potentially_feasible=False
                        break
                    elif x in V:
                        continue
                    model.Add(variables[x] <= variables[y] + c)
                if not potentially_feasible:
                    continue
                solver = cp_model.CpSolver()
                status = solver.Solve(model)
                if status == cp_model.OPTIMAL:
                    assignment= {v.name: solver.Value(variables[v]) for v in self.formal_variables}
                    for infs in V:
                        assignment[infs.name]=-np.inf
                    return assignment
        raise Exception("No solution found")


def ternary_max_atom_to_dnf(system: max_atom.TernaryMaxAtomSystem) -> MaxAtomCpModel:
    """
    Convert a ternary max-atom system to a disjunctive normal form (DNF)
    """
    model = MaxAtomCpModel()
    for x, y, z, c in system.constraints:
        model.addMaxAtom(x, [y, z], c)
    return model


def min_max_atom_to_dnf(system: max_atom.MinMaxSystem) -> MinMaxCpModel:
    """
    Convert a ternary max-atom system to a disjunctive normal form (DNF)
    """
    model = MinMaxCpModel()
    for op, x, y, c in system.constraints:
        if op == "max":
            model.addMaxAtom(x, y, c)
        else:
            model.addMinAtom(x, y, c)
    return model


if __name__ == "__main__":
    G = mpg.mpg_from_file("data/test01.in", ignore_header=1)
    G.closure()
    model = min_max_atom_to_dnf(G.as_min_max_system())
    assignment = model.solve()

    print(assignment)
