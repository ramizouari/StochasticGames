import pytest
from . import mpg
from ..graph import random_graph

class MPGData:
    def __init__(self, game:mpg.MeanPayoffGraph, index:int, turn:int):
        self.game=game
        self.index=index
        self.turn=turn

    def __iter__(self):
        yield self.game
        yield self.index
        yield self.turn
@pytest.fixture(scope='class', params=range(8))
def special_mpg(request)->MPGData:
    n=request.param
    game=mpg.MeanPayoffGraph()
    index=0
    turn=mpg.MeanPayoffGraph.player0
    match n:
        case 0:
            game.add_edge(0,1,0)
            game.add_edge(1,0,0)
        case 1:
            game.add_edge(0,1,0)
            game.add_edge(1,0,1)
        case 2:
            game.add_edge(0,1,-1)
            game.add_edge(1,0,0)
        case 3|4:
            game.add_weighted_edges_from([(1, 2, 5),
                                 (2, 3, -7),
                                (3, 7, 0),
                                 (3, 6, 5),
                                 (6, 1, -3),
                                 (1, 4, 4),
                                 (4, 5, -3),
                                 (5, 6, 3),
                                 (5, 7, 0),
                                 (7, 1, 0),
                                 (0, 1, 5)])
            if n==4:
                turn=mpg.MeanPayoffGraph.player1
        case _:
            if n%2==0:
                game= random_graph.gnm_random_mpg(n=10,m=20,seed=n,method="fast",loops=True,distribution="integers",low=-7,high=8)
            else:
                game= random_graph.gnp_random_mpg(n=10,p=0.5,seed=n,method="fast",loops=True,distribution="integers",low=-7,high=8)
    return MPGData(game, index, turn)

class TestMPG:
    def test_bipartite(self, special_mpg:MPGData):
        game, index, turn=special_mpg
        BP=game.as_bipartite()

        phi={}
        for z in BP.nodes:
            u,turn=z
            if not u in phi:
                phi[u]=set()
            phi[u]|={z}
        for u in phi:
            assert len(phi[u])==2
        assert len(phi)==len(game.nodes)

        for x,y in BP.edges:
            u,p=x
            v,q=y
            assert(p!=q)

    def test_dual(self, special_mpg:MPGData):
        game, index, turn=special_mpg
        dual=game.dual()
        for x,y in game.edges:
            assert (x,y) in dual.edges
            assert dual.edges[x,y]["weight"]==-game.edges[x,y]["weight"]

    def test_dual_dual(self,special_mpg):
        game, index, turn=special_mpg
        dualdual=game.dual().dual()
        expectedResult = {(x,y,game.edges[x,y]["weight"]) for x,y in game.edges}
        result={(x,y,dualdual.edges[x,y]["weight"]) for x,y in dualdual.edges}
        assert result==expectedResult

