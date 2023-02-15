import graph as g


G=g.LabeledGraph(4)
G.addEdge((0,1,5))
G.addEdge((1,3,3))
G.addEdge((3,2,-8))
G.addEdge((2,0,-4))
print(g.BellmanFordAlgorithm(G,0))