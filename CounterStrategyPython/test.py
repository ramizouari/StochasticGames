import graph as g


import graph as g
with open("data/test01.in",'r') as file:
    V,E=map(int,file.readline().rstrip().split())
    G=g.LabeledGraph(V)
    for k in range(E):
        G.addEdge(map(int,file.readline().rstrip().split()))
    psi=list(map(int,file.readline().rstrip().split()))
G=g.read_from_text_file("data/test01.in",graph_type="auto")
print(g.counterStrategy(G,psi,method="bellman-ford"))
