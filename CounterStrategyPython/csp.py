import abc
from typing import Dict, List, Any


# This class represents a variable in a constraint satisfaction problem
class Variable:
    default_name: str = "X"

    def __init__(self, id = None, name: str = None):
        self.id = id
        if name is None:
            self.name = f"X[{self.id}]"
        else:
            self.name=name
    pass

    def __repr__(self):
        return self.name

    def __eq__(self, other):
        if type(self) != type(other):
            return False
        return self.id == other.id

    def __hash__(self):
        return self.id.__hash__()


# This class represents a constraint satisfaction problem
class ConstraintSatisfactionProblem(abc.ABC):

    def __init__(self):
        self.variables = set()

    @abc.abstractmethod
    def solve(self,*args,**kwargs) -> Dict[Variable, List[Any]]:
        pass

    def satisfiable(self, L=None, R=None)-> bool:
        admissible_values = self.solve(L, R)
        return all(len(admissible_values[u]) > 0 for u in self.variables)
