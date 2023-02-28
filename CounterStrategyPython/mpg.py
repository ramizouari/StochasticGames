import networkx as nx
import max_atom as ma



class MeanPayoffGraph(nx.DiGraph):
    def __init__(self):
        super().__init__()
        self.bipartite=False

    def closure(self):
        for u in self.nodes:
            if not self.out_edges(u):
                self.add_edge(u, u, weight=0)

    def add_edge(self, u, v, weight,**kwattr):
        super().add_edge(u, v, weight=weight,label=weight,**kwattr)

    def as_bipartite(self) -> "MeanPayoffGraph":
        if not self.bipartite:
            G = MeanPayoffGraph()
            for u in self.nodes:
                for v in self.succ[u]:
                    L= self[u][v]["weight"]
                    G.add_edge((u,0), (v,1), weight=L)
                    G.add_edge((u,1),(v,0),weight=L)
            G.bipartite=True
        else:
            G=self
        return G

    def as_min_max_system(self) -> ma.MinMaxSystem:
        S=ma.MinMaxSystem()
        BG=self.as_bipartite()
        for u in BG.nodes:
            _,p=u
            V=[v for v in BG.succ[u]]
            L=[BG[u][v]["weight"] for v in V]
            namer=lambda v: f"@{v[0]}P{v[1]}"
            S.add_constraint("min" if p==1 else "max",ma.Variable(id=u,name=namer(u)),[ma.Variable(id=v,name=namer(v)) for v in V],L)
        return S


if __name__=="__main__":
    G=MeanPayoffGraph()
    G.add_edge(1,2,1)
    G.add_edge(2,3,2)
    G.add_edge(3,1,3)
    G.closure()
    print(G.edges(data=True))
    print(G.as_min_max_system())