import queue
from typing import Tuple, List, Dict, Any, Union

import numpy as np

import csp
from csp import ConstraintSatisfactionProblem, Variable


# This class represents a variable in a ternary max atom system
# A ternary max atom system is a system of equations of the form
# x <= max(y,z) + c
class TernaryMaxAtomSystem(ConstraintSatisfactionProblem):
    """
    A ternary max atom system is a system of equations of the form:

    - **x_1 <= max(y_1,z_1) + c_1**
    - **x_2 <= max(y_2,z_2) + c_2**
    - ...
    - **x_n <= max(y_n,z_n) + c_n**

    where **x_i**, **y_i**, **z_i** are variables and **c_i** is a constant.
    """
    DEFAULT_METHOD: str = "ACO"

    def __init__(self):
        """
        Initialize the system
        """
        super().__init__()
        self.constraints = []
        self.lhs_constraints = {}
        self.rhs_constraints = {}
        self.adjacency_list = {}
        self.assignment = None

    def add_constraint(self, x, y, z, c):
        """
        Add a constraint to the system
        :param x: The variable on the left hand side of the constraint
        :param y: The first variable on the right hand side of the constraint
        :param z: The second variable on the right hand side of the constraint
        :param c: The constant on the right hand side of the constraint
        :return: None
        """
        self.constraints.append([x, y, z, c])
        self.variables.add(x)
        self.variables.add(y)
        self.variables.add(z)
        V = (x, y, z)
        for order in range(3):
            if V[order] not in self.adjacency_list:
                self.adjacency_list[V[order]] = set()
            if V[order] not in self.lhs_constraints:
                self.lhs_constraints[V[order]] = []
            if V[order] not in self.rhs_constraints:
                self.rhs_constraints[V[order]] = []
            self.adjacency_list[V[order]].add(V[(order + 1) % 3])
            self.adjacency_list[V[order]].add(V[(order + 2) % 3])
        self.lhs_constraints[x].append((y, z, c))
        self.rhs_constraints[y].append((x, z, c))
        self.rhs_constraints[z].append((x, y, c))
        pass

    def __repr__(self):
        endline = "\n\t"
        return f"""System of {len(self.constraints)} equations:
\t{endline.join([f"{x} <= max({y},{z}) + {c}" for x, y, z, c in self.constraints])}
"""

    def _repr_latex_(self):
        endline = "\\\\"
        output=endline.join(sorted([f"{x} &\leq {y} + {c}" for x, y, z, c in self.constraints]))
        return fR"""System of {len(self.constraints)} equations:
        \begin{{align*}}
{output}
\end{{align*}}"""

    def _consistency(self, constraint: Tuple[int, int, int, int], admissible_values: dict, Q: queue.Queue,
                     order: int = None):
        """
        Check the consistency of the system
        :param constraint: The constraint to check
        :param admissible_values: The admissible values for each variable
        :param Q: The queue of variables to check
        :param order: The order in which to check the constraint
        :return: None
        """
        if order is None:
            # Check the consistency of the constraint for each order
            for k in range(3):
                # Check the consistency of the constraint for the order k
                self._consistency(constraint, admissible_values, Q, order=k)
        else:
            # Check the consistency of the constraint for the order k
            # P is the order in which the variables appear in the constraint
            P = [(order + k) % 3 for k in range(3)]
            # I is the order in which the variables appear in the tuple H
            I = [(-order + k) % 3 for k in range(3)]
            # c is the constant of the constraint
            c = constraint[-1]
            # List of values to delete from the admissible values of the variable x
            toDelete = []
            for s in admissible_values[constraint[P[0]]]:
                admissible = False
                for t in admissible_values[constraint[P[1]]]:
                    for r in admissible_values[constraint[P[2]]]:
                        H = (s, t, r)
                        if H[I[0]] <= max(H[I[1]], H[I[2]]) + c:
                            admissible = True
                        if admissible:
                            break
                    if admissible:
                        break
                if not admissible:
                    toDelete.append(s)
            for s in toDelete:
                for v in self.adjacency_list[constraint[P[0]]]:
                    Q.put(v)
                admissible_values[constraint[P[0]]].remove(s)
                pass

    def _arc_consistency_optimized(self, L=None, R=None) -> Dict[Variable, int]:
        # init AC lists
        K = np.sum(np.fromiter((np.abs(c) for x, y, z, c in self.constraints), dtype=int))
        if L is None:
            L = -K
        if R is None:
            R = K
        assignment = {u: R for u in self.variables}
        # prepare rhs list
        rhs_constraints = self.rhs_constraints

        # init queue
        Q = queue.Queue()
        for u in self.variables:
            Q.put(u)
        while not Q.empty():
            v = Q.get()
            for u, w, c in rhs_constraints[v]:
                newValue = max(assignment[v], assignment[w]) + c
                if newValue < L:
                    newValue = -np.inf
                if newValue < assignment[u]:
                    assignment[u] = newValue
                    Q.put(u)
        return assignment

    def solve(self, L=None, R=None, method=DEFAULT_METHOD) -> Union[Dict[Variable, Any], None]:
        """
        Solve the system of equations
        :param L: The lower bound of the admissible values
        :param R: The upper bound of the admissible values
        :param method: The method to use to solve the system
        :return: A dictionary mapping each variable to an admissible value, or None if the system is inconsistent
        """
        match method:
            case "AC" | "arc_consistency" | "arc-consistency":
                # Sum of all constants
                K = np.sum(np.fromiter((np.abs(c) for x, y, z, c in self.constraints), dtype=int))
                if L is None:
                    L = -K
                if R is None:
                    R = K
                admissible_values = {u: set(range(L, R + 1)) for u in self.variables}
                for u in self.variables:
                    admissible_values[u].add(-np.inf)
                Q = queue.Queue()
                for u in self.variables:
                    Q.put(u)
                while not Q.empty():
                    u = Q.get()
                    for v, w, c in self.lhs_constraints[u]:
                        self._consistency(constraint=(u, v, w, c), admissible_values=admissible_values, Q=Q)
                # Applies the max polymorphism to the admissible values
                self.assignment = {u: max(admissible_values[u]) for u in self.variables}

            case "ACO" | "arc_consistency_optimized" | "arc-consistency-optimized":
                self.assignment = self._arc_consistency_optimized(L=L, R=R)
        return self.assignment

    # Return True if there is an assignment on which every value is finite
    def satisfiable(self, L=None, R=None, method=DEFAULT_METHOD) -> bool:
        """
        Return True if there is an assignment on which every value is finite
        :param L: The lower bound of the admissible values
        :param R: The upper bound of the admissible values
        :param method: The method to use to solve the system
        :return: True if there is an assignment on which every value is finite
        """
        if self.assignment is None:
            self.solve(L, R, method=method)
        return all(map(lambda u: u > -np.inf, self.assignment))


# This class represents a variable in a max atom system
# A max atom system is a system of equations of the form
# x <= max(Y) + c
# where Y is a set of variables
class MaxAtomSystem(ConstraintSatisfactionProblem):
    """
    This class represents a variable in a max atom system
    A max atom system is a system of equations of the form:

    - **x1 <= max(Y1) + c1**

    - **x2 <= max(Y2) + c2**

    - ...

    - **xn <= max(Yn) + cn**

    Where **x1,..,xn** are variables, **Y1,..,Yn** are set of variables and **c1,..,cn** are constants.
    """
    DEFAULT_METHOD: str = TernaryMaxAtomSystem.DEFAULT_METHOD

    def __init__(self):
        """
        Initialize the max atom system
        """
        super().__init__()
        self.constraints = []
        self.mapper = {}
        self.equivalent_system = TernaryMaxAtomSystem()
        self.variable_generator = csp.VariableGenerator(name="P")

    def add_constraint(self, x, Y, c):
        """
        Add a constraint to the system
        :param x: The variable on the left-hand side of the constraint
        :param Y: The set of variables on the right-hand side of the constraint
        :param c: The constant on the right-hand side of the constraint
        :return: None
        """
        if len(Y) == 0:
            raise RuntimeError("right-hand side contains 0 variables")
        self.constraints.append([x, Y, c])
        _Y = Y.copy()
        while len(_Y) > 2:
            y = _Y.pop()
            z = _Y.pop()
            if (y, z) not in self.mapper:
                variable=self.variable_generator()
                self.mapper[(y, z)] = variable
                self.mapper[(z, y)] = variable
                self.variables.add(variable)
            w = self.mapper[(y, z)] # w is the variable that represents max(y,z)
            _Y.append(w)
            self.equivalent_system.add_constraint(w, y, z, 0)
        # We have m=length(_Y) is in {1,2}
        # In any case, this constraint will work for both cases
        self.equivalent_system.add_constraint(x, _Y[0], _Y[-1], c)
        self.variables.add(x)
        self.variables.update(Y)

    def solve(self, L=None, R=None, method=DEFAULT_METHOD) -> Dict[Variable, List[Any]]:
        """
        Solve the system of equations
        :param L: The lower bound of the admissible values
        :param R: The upper bound of the admissible values
        :param method: The method to use to solve the system
        :return: A dictionary mapping each variable to an admissible value, or None if the system is inconsistent
        """
        S_augmented = self.equivalent_system.solve(L, R, method=method)
        return {u: S_augmented[u] for u in self.variables}

    def __repr__(self):
        endline = "\n\t"
        output = []
        for x, Y, c in self.constraints:
            args = ",".join([f"{y}" for y in Y])
            output.append(f"{x} <= max({args}) + {c}")
        return f"System of {len(self.constraints)} max constraints:\n\t" + endline.join(output)

    def _repr_latex_(self):
        endline = "\\\\\n\t"
        output = []
        for x, Y, c in self.constraints:
            args = ",".join([f"{y}" for y in Y])
            output.append(f"{x} & \\leq \\max({args}) + {c}")
        return fR"""System of {len(self.constraints)} max constraints:
        \begin{{align*}}
        {endline.join(sorted(output))}
        \end{{align*}}
        """

# This class represents a variable in a min max system
# A min max system is a system of equations of the form
# x <= op(y1+c1,y2+c2,...,yn+cn)
# where Y=[y1,y2,...,yn] is a set of variables
# and C=[c1,c2,...,cn] is a set of constants
class MinMaxSystem(ConstraintSatisfactionProblem):
    """
    This class represents a variable in a min max system
    A min max system is a system of equations of the form:

    - **x1 <= OP1(y11+c11,y12+c12,...,y1n+c1n)**

    - **x2 <= OP2(y21+c21,y22+c22,...,y2n+c2n)**

    - ...

    - **xn <= OPn(yn1+cn1,yn2+cn2,...,ynn+cn)**

    Where **x1,..,xn** are variables, **yij** are variables and **cij** are constants, and **OP1,..,OPn** are either **min** or **max**.
    """
    DEFAULT_METHOD: str = MaxAtomSystem.DEFAULT_METHOD

    def __init__(self):
        """
        Initialize the min max system
        """
        super().__init__()
        self.constraints = []
        self.mapper = {}
        self.equivalent_system = MaxAtomSystem()
        self.variable_generator = csp.VariableGenerator(name="S")

    def add_constraint(self, op, x, Y, C):
        """
        Add a constraint to the system
        :param op: The operator of the constraint. It can be either "min" or "max"
        :param x: The variable on the left-hand side of the constraint
        :param Y: The list of variables on the right-hand side of the constraint
        :param C: The list of constants on the right-hand side of the constraint
        :return: None
        """
        if op not in ["min", "max"]:
            raise RuntimeError("operator is not min or max")
        # add variables to the set of variables
        self.variables.add(x)
        self.variables.update(Y)
        # add constraint to the list of constraints
        self.constraints.append([op, x, Y, C])
        # if op is min, then we add the constraints x <= y1+c , x <= y2+c , ... , x <= yn+cn
        if op == "min":
            for y, c in zip(Y, C):
                self.equivalent_system.add_constraint(x, [y, y], c)
            pass
        else:
            Z = []
            # if op is max, then we add the constraints x <= max(z1,z2,...,zn),
            # z1 <= y1+c1, z2 <= y2+c2, ... , zn <= yn+cn
            for y, c in zip(Y, C):
                if (y, c) not in self.mapper:
                    self.mapper[(y, c)] = self.variable_generator()
                z = self.mapper[(y, c)]
                Z.append(z)
                self.equivalent_system.add_constraint(z, [y, y], c)
            self.equivalent_system.add_constraint(x, Z, 0)
            pass

    def solve(self, L=None, R=None,  method=DEFAULT_METHOD) -> Dict[
        Variable, List[Any]]:
        """
        Solve the system of equations
        :param L: The lower bound of the admissible values
        :param R: The upper bound of the admissible values
        :param method: The method to use to solve the system
        :return: A dictionary mapping each variable to an admissible value, or None if the system is inconsistent
        """
        S_augmented = self.equivalent_system.solve(L, R, method=method)
        return {u: S_augmented[u] for u in self.variables}

    def __repr__(self):
        endline = "\n\t"
        output = []
        for op, x, Y, C in self.constraints:
            args = ",".join([f"{y}+{c}" for y, c in zip(Y, C)])
            output.append(f"{x} <= {op}({args})")
        return f"System of {len(self.constraints)} min-max constraints:\n\t" + endline.join(output)

    def _repr_latex_(self):
        endline = "\\\\\n"
        output = []
        for op, x, Y, C in self.constraints:
            args = ",".join([f"{y}+{c}" for y, c in zip(Y, C)])
            output.append(f"{x} & \\leq \\{op}({args})")
        return fR"""System of {len(self.constraints)} min-max constraints:
        \begin{{align*}}
        {endline.join(sorted(output))} 
        \end{{align*}}"""


if __name__ == "__main__":
    system = TernaryMaxAtomSystem()
    system.add_constraint(Variable(1), Variable(1), Variable(1), -1)
    print(system.solve(L=-2, R=2, method="ACO"))
    print(system.solve(L=-2, R=2, method="AC"))
    system = TernaryMaxAtomSystem()
    system.add_constraint(Variable(1), Variable(1), Variable(2), -1)
    system.add_constraint(Variable(2), Variable(2), Variable(1), -1)
    print(system.solve(L=-2, R=2, method="ACO"))
    print(system.solve(L=-2, R=2, method="AC"))
