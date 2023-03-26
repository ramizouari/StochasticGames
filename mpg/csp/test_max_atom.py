import numpy as np

from . import max_atom
import pytest

@pytest.fixture(scope='class', params=[[(0,0,0,-1)],
                                  [(0,0,0,0)],
                                  [(0,0,0,1)],
                                  [(0,1,1,-1),(1,0,0,-1)],])
def ternary_constraints(request):
    return request.param


@pytest.mark.usefixtures("ternary_constraints")
class TestTernaryMaxAtom:

    def add_constraints(self,ternary_constraints):
        self.system=max_atom.TernaryMaxAtomSystem()
        self.constraints=ternary_constraints
        for x,y,z,w in ternary_constraints:
            self.system.add_constraint(x,y,z,w)
        return self.system
    def test_constraints(self,ternary_constraints):
        self.add_constraints(ternary_constraints)
        assert {(x,y,z,w) for x,y,z,w in self.system.constraints}=={(x,y,z,w) for x,y,z,w in self.constraints}

    def test_variables(self,ternary_constraints):
        system=self.add_constraints(ternary_constraints)
        assert system.variables=={x for x,y,z,w in self.constraints} | {y for x,y,z,w in self.constraints} | {z for x,y,z,w in self.constraints}
    def test_arc_consistency_solution(self,ternary_constraints):
        system=self.add_constraints(ternary_constraints)
        assignment=system.solve()
        for x,y,z,w in self.constraints:
            assert assignment[x]<=assignment[y]+assignment[z]+w

    def test_arc_consistency_maximality(self,ternary_constraints):
        system=self.add_constraints(ternary_constraints)
        assignment=system.solve()
        fixedVariable=next(iter(system.variables))
        for x in system.variables:
            if x==fixedVariable:
                continue
            oldX=assignment[x]
            assignment[x]+=1
            if assignment[x]==oldX:
                continue
            satisfied=True
            for x,y,z,w in self.constraints:
                if assignment[x]>assignment[y]+assignment[z]+w:
                    satisfied=False
                    break
            assert not satisfied
            assignment[x]-=1


@pytest.fixture(scope='class', params=[[(0,[0,0],-1)],
                                  [(0,[0],0)],
                                  [(0,[0,0,0],1)],
                                  [(0,[1],-1),(1,[0],-1)],])
def max_atom_constraints(request):
    return request.param

@pytest.mark.usefixtures("max_atom_constraints")
class TestMaxAtom:
    def add_constraints(self,max_atom_constraints):
        self.system=max_atom.MaxAtomSystem()
        self.constraints=max_atom_constraints
        for x,Y,w in max_atom_constraints:
            self.system.add_constraint(x,Y,w)
        return self.system
    def test_constraints(self,max_atom_constraints):
        self.add_constraints(max_atom_constraints)
        assert {(x,*Y,w) for x,Y,w in self.system.constraints}=={(x,*Y,w) for x,Y,w in self.constraints}

    def test_variables(self,max_atom_constraints):
        self.add_constraints(max_atom_constraints)
        expectedVariables={x for x,Y,w in self.constraints}
        for Y in (Y for x,Y,w in self.constraints):
            expectedVariables|=set(Y)

        assert self.system.variables==expectedVariables
    def test_arc_consistency_solution(self,max_atom_constraints):
        self.add_constraints(max_atom_constraints)
        assignment=self.system.solve()
        for x,Y,w in self.constraints:
            assert assignment[x]<=max(assignment[y] for y in Y)+w

    def test_arc_consistency_maximality(self,max_atom_constraints):
        self.add_constraints(max_atom_constraints)
        assignment=self.system.solve()
        fixedVariable=next(iter(self.system.variables))
        for x in self.system.variables:
            if x==fixedVariable:
                continue
            oldX=assignment[x]
            assignment[x]+=1
            if assignment[x]==oldX:
                continue
            satisfied=True
            for x,y,w in self.constraints:
                if assignment[x]>assignment[y]+w:
                    satisfied=False
                    break
            assert not satisfied
            assignment[x]-=1
