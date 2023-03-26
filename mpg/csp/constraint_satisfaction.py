import abc
from typing import Dict, List, Any


# This class represents a variable in a constraint satisfaction problem
class Variable:
    """
    A variable in a constraint satisfaction problem.
    It has a name and an id.
    """
    default_name: str = "X"
    """
    The default name of a nameless variable.
    """

    def __init__(self, id=None, name: str = None):
        """
        Initialise a variable.
        :param id: The id of the variable. If None, a new id is generated. id must be hashable.
        :param name: The name of the variable. If None, the name is set by default to "X[id]".
        """
        self.id = id
        if name is None:
            self.name = f"X[{self.id}]"
        else:
            self.name = name

    pass

    def __repr__(self):
        return self.name

    def __eq__(self, other):
        if type(self) != type(other):
            return False
        return self.id == other.id

    def __hash__(self):
        return self.id.__hash__()


class VariableGenerator:
    """
    A generator of variables.
    """

    def __init__(self, name=None):
        self.id = 0
        self.name = name
        if name is None:
            self.name = Variable.default_name

    def variable_name(self, id):
        return f"{self.name}[{id}]"


    def __call__(self) -> Variable:
        """
        Generate a new variable.
        :param name: The name of the variable. If None, the name is set by default to "X[id]".
        :return: The new variable.
        """
        self.id += 1
        return Variable((self.id,self), self.variable_name(self.id))

    def __eq__(self, other):
        return self is other

    def __hash__(self):
        return id(self)

# This class represents a constraint satisfaction problem
class ConstraintSatisfactionProblem(abc.ABC):
    """
    A constraint satisfaction problem.
    """

    def __init__(self):
        self.variables = set()

    @abc.abstractmethod
    def solve(self, *args, **kwargs) -> Dict[Variable, Any]:
        """
        Solve the constraint satisfaction problem.
        :param args: Arguments to the solver.
        :param kwargs: Keyword arguments to the solver.
        :return: A dictionary mapping each variable to an admissible value.
        """
        pass

    def satisfiable(self, L=None, R=None) -> bool:
        """
        Check if the constraint satisfaction problem is satisfiable.
        :param L: The lower bound on the admissible values.
        :param R: The upper bound on the admissible values.
        :return: True if the constraint satisfaction problem is satisfiable, False otherwise.
        """
        admissible_values = self.solve(L, R)
        return all(len(admissible_values[u]) > 0 for u in self.variables)

    def admissible(self, assignment: Dict[Variable, Any]) -> bool:
        pass
