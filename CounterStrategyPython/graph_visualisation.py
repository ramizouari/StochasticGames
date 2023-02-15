import graph as g
from yfiles_jupyter_graphs import GraphWidget

def visualise_graph(G):
    W=GraphWidget()
    W.nodes=[{"id":u,"properties":{"label":u}} for u in G.vertices()]
    W.edges=[]
    W.directed=True
    k=0
    for u in range(G.V):
        for (v,*L) in G.adjacencyList[u]:
            properties={}
            if len(L)==1:
                properties["label"]=L[0]
            elif len(L)>0:
                properties["label"]=L
            W.edges.append({"id":k,"start":u,"end":v,"properties":properties})
            k+=1
    return W

class VisualGraph(GraphWidget):
    def __init__(self,graphClass,V):
        super().__init__()
        self.graph=graphClass(V)
        self.nodes=[{"id":u,"properties":{"label":u}} for u in self.graph.vertices()]
        self.edges=[]
        self.directed=True
    
    def addEdge(self,edge,**kwargs):
        (u,v,*L)=edge
        edgeMetadata=kwargs.copy()
        if len(L)>1:
            edgeMetadata["label"]=L
        elif len(L)==1:
            edgeMetadata["label"]=L[0]
        self.edges.append({"id":self.graph.E,"start":u,"end":v,"properties":edgeMetadata})
        self.graph.addEdge(edge)

        
