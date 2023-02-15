class UnionFind:
    def __init__(self, n: int):
        self.n = n
        self.parent = [i for i in range(n)]
        self.rank = [0 for i in range(n)]

    def representative(self, a) -> int:
        if self.parent[a] == a:
            return a
        else:
            self.parent[a] = self.representative(self.parent[a])
            return self.parent[a]

    def equivalent(self, a: int, b: int) -> bool:
        return self.representative(a) == self.representative(b)

    def connect(self, a: int, b: int):
        u = self.representative(a)
        v = self.representative(b)
        if self.rank[u] < self.rank[v]:
            self.parent[u] = self.parent[v]
        elif self.rank[u] > self.rank[v]:
            self.parent[v] = self.parent[u]
        else:
            self.parent[u] = self.parent[v]
            self.rank[v] += 1
        pass
