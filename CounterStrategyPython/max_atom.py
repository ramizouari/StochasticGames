import abc
import queue
from typing import Tuple, List, Set, Dict, Any

import numpy as np

from csp import ConstraintSatisfactionProblem, Variable


# This class represents a variable in a ternary max atom system
# A ternary max atom system is a system of equations of the form
# x <= max(y,z) + c
class TernaryMaxAtomSystem(ConstraintSatisfactionProblem):
    def __init__(self):
        super().__init__()
        self.constraints = []
        self.lhs_constraints={}
        self.adjacency_list={}

    def add_constraint(self, x, y, z, c):
        self.constraints.append([x,y,z,c])
        self.variables.add(x)
        self.variables.add(y)
        self.variables.add(z)
        V = (x, y, z)
        for order in range(3):
            if V[order] not in self.adjacency_list:
                self.adjacency_list[V[order]]=set()
            if V[order] not in self.lhs_constraints:
                self.lhs_constraints[V[order]]=[]
            self.adjacency_list[V[order]].add(V[(order+1)%3])
            self.adjacency_list[V[order]].add(V[(order+2)%3])
        self.lhs_constraints[x].append((y,z,c))
        pass

    def __repr__(self):
        endline = "\n\t"
        return f"""System of {len(self.constraints)} equations:
\t{endline.join([f"{x} <= max({y},{z}) + {c}" for x,y,z,c in self.constraints])}
"""

    def _consistency(self,constraint:Tuple[int,int,int,int],admissible_values:dict,Q:queue.Queue,order:int=None):
        if order is None:
            for k in range(3):
                self._consistency(constraint,admissible_values,Q,order=k)
        else:
            P=[(order+k)%3 for k in range(3)]
            I=[(-order+k)%3 for k in range(3)]
            c=constraint[-1]
            toDelete=[]
            for s in admissible_values[constraint[P[0]]]:
                admissible=False
                for t in admissible_values[constraint[P[1]]]:
                    for r in admissible_values[constraint[P[2]]]:
                        H=(s,t,r)
                        if H[I[0]] <= max(H[I[1]],H[I[2]])+c:
                            admissible=True
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

    def solve(self,L=None,R=None) -> Dict[Variable, List[Any]]:
        #Sum of all constants
        K= np.sum(np.fromiter((np.abs(c) for x,y,z,c in self.constraints),dtype=int))
        if L is None:
            L=-K
        if R is None:
            R=K
        admissible_values={u:set(range(L,R+1)) for u in self.variables}
        Q=queue.Queue()
        for u in self.variables:
            Q.put(u)
        while not Q.empty():
            u=Q.get()
            for v,w,c in self.lhs_constraints[u]:
                self._consistency(constraint=(u,v,w,c),admissible_values=admissible_values,Q=Q)
        return admissible_values


# This class represents a variable in a max atom system
# A max atom system is a system of equations of the form
# x <= max(Y) + c
# where Y is a set of variables
class MaxAtomSystem(ConstraintSatisfactionProblem):
    def __init__(self):
        super().__init__()
        self.constraints = []
        self.mapper={}
        self.equivalent_system=TernaryMaxAtomSystem()
        self.counter=0

    def add_constraint(self,x,Y,c):
        if len(Y) == 0:
            raise RuntimeError("right-hand side contains 0 variables")
        self.constraints.append([x,Y,c])
        _Y=Y.copy()
        while len(_Y) > 2:
            y=_Y.pop()
            z=_Y.pop()
            if (y,z) not in self.mapper:
                self.mapper[(y,z)]=Variable(id=f"#{self.counter}")
                self.mapper[(z,y)]=Variable(id=f"#{self.counter}")
                self.counter+=1
            w=self.mapper[(y,z)]
            _Y.append(w)
            self.equivalent_system.add_constraint(y,z,w,0)
        #We have m=length(_Y) is in {1,2}
        #In any case, this constraint will work for both cases
        self.equivalent_system.add_constraint(x,_Y[0],_Y[-1],c)
        self.variables.add(x)
        self.variables.update(Y)
    def solve(self,L=None,R=None)  -> Dict[Variable, List[Any]]:
        S_augmented=self.equivalent_system.solve(L,R)
        return {u:S_augmented[u] for u in self.variables}


# This class represents a variable in a min max system
# A min max system is a system of equations of the form
# x <= op(y1+c1,y2+c2,...,yn+cn)
# where Y=[y1,y2,...,yn] is a set of variables
# and C=[c1,c2,...,cn] is a set of constants
class MinMaxSystem(ConstraintSatisfactionProblem):
    def __init__(self):
        super().__init__()
        self.constraints = []
        self.mapper={}
        self.equivalent_system=MaxAtomSystem()
        self.counter=0

    def add_constraint(self,op,x,Y,C):
        if op not in ["min", "max"]:
            raise RuntimeError("operator is not min or max")

        self.variables.add(x)
        self.variables.update(Y)
        self.constraints.append([op,x,Y,C])
        if op == "min":
            for y,c in zip(Y,C):
                self.equivalent_system.add_constraint(x,[y,y],c)
            pass
        else:
            Z=[]
            for y,c in zip(Y,C):
                if (y,c) not in self.mapper:
                    self.mapper[(y,c)]=Variable(self.counter)
                    self.counter+=1
                z=self.mapper[(y,c)]
                Z.append(z)
                self.equivalent_system.add_constraint(y,[z,z],c)
            self.equivalent_system.add_constraint(x,Z,0)
            pass

    def solve(self,L=None,R=None) -> Dict[Variable, List[Any]]:
        S_augmented = self.equivalent_system.solve(L, R)
        return {u: S_augmented[u] for u in self.variables}

    def __repr__(self):
        endline = "\n\t"
        output=[]
        for op,x,Y,C in self.constraints:
            args=",".join([f"{y}+{c}" for y,c in zip(Y,C)])
            output.append(f"{x} <= {op}({args})")
        return f"System of {len(self.constraints)} min-max constraints:\n\t"+endline.join(output)


if __name__=="__main__":
    system = MaxAtomSystem()
    system.add_constraint(Variable(1),[Variable(1),Variable(1)],0)
    system.add_constraint(Variable(1),[Variable(1),Variable(1),Variable(2)],-1)
    print(system.solve(L=-2,R=2))

    S=MinMaxSystem()
    x=Variable(1)
    y=Variable(2)
    S.add_constraint("min",x,[y],[0])
    S.add_constraint("max",y,[x],[0])
    print(S.solve(L=-2,R=2))