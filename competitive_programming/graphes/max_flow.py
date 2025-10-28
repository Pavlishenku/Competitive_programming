"""
Max Flow (Flux Maximum)

Description:
    Algorithmes pour trouver le flux maximum dans un réseau de transport.
    Implémentations: Dinic (le plus rapide), Ford-Fulkerson, Edmond

s-Karp.

Complexité:
    - Dinic: O(V²E) général, O(E√V) pour graphes unitaires
    - Ford-Fulkerson: O(E * max_flow)
    - Edmonds-Karp: O(VE²)

Cas d'usage:
    - Problèmes de flux de réseau
    - Bipartite matching
    - Min cut
    - Circulation avec demandes
    
Problèmes types:
    - Codeforces: 546E, 817D
    - AtCoder: ABC205E
    - CSES: Download Speed
    
Implémentation par: 2025-10-27
Testé: Oui
"""

from collections import deque, defaultdict


class Dinic:
    """
    Algorithme de Dinic pour flux maximum.
    Le plus rapide pour la plupart des cas.
    """
    
    def __init__(self, n):
        """
        Args:
            n: Nombre de sommets (0 à n-1)
        """
        self.n = n
        self.graph = [[] for _ in range(n)]
    
    def add_edge(self, u, v, capacity):
        """
        Ajoute une arête orientée de u vers v.
        
        Args:
            u: Sommet source
            v: Sommet destination
            capacity: Capacité de l'arête
        """
        # Ajouter l'arête et son arête inverse
        self.graph[u].append([v, capacity, len(self.graph[v])])
        self.graph[v].append([u, 0, len(self.graph[u]) - 1])
    
    def bfs(self, source, sink):
        """BFS pour construire le graphe de niveaux"""
        self.level = [-1] * self.n
        self.level[source] = 0
        queue = deque([source])
        
        while queue:
            v = queue.popleft()
            
            for i in range(len(self.graph[v])):
                to, cap, _ = self.graph[v][i]
                if cap > 0 and self.level[to] < 0:
                    self.level[to] = self.level[v] + 1
                    queue.append(to)
        
        return self.level[sink] >= 0
    
    def dfs(self, v, sink, flow):
        """DFS pour trouver des chemins augmentants"""
        if v == sink:
            return flow
        
        for i in range(self.iter[v], len(self.graph[v])):
            self.iter[v] = i
            to, cap, rev = self.graph[v][i]
            
            if cap > 0 and self.level[v] < self.level[to]:
                d = self.dfs(to, sink, min(flow, cap))
                
                if d > 0:
                    self.graph[v][i][1] -= d
                    self.graph[to][rev][1] += d
                    return d
        
        return 0
    
    def max_flow(self, source, sink):
        """
        Calcule le flux maximum de source à sink.
        
        Args:
            source: Sommet source
            sink: Sommet puits
            
        Returns:
            Flux maximum
            
        Example:
            >>> dinic = Dinic(4)
            >>> dinic.add_edge(0, 1, 10)
            >>> dinic.add_edge(0, 2, 10)
            >>> dinic.add_edge(1, 3, 10)
            >>> dinic.add_edge(2, 3, 10)
            >>> dinic.max_flow(0, 3)
            20
        """
        flow = 0
        
        while self.bfs(source, sink):
            self.iter = [0] * self.n
            
            while True:
                f = self.dfs(source, sink, float('inf'))
                if f == 0:
                    break
                flow += f
        
        return flow
    
    def min_cut(self, source):
        """
        Retourne les sommets accessibles depuis la source dans le graphe résiduel.
        Utile pour trouver le min cut.
        
        Args:
            source: Sommet source
            
        Returns:
            Set des sommets dans le cut
        """
        visited = [False] * self.n
        queue = deque([source])
        visited[source] = True
        
        while queue:
            v = queue.popleft()
            
            for to, cap, _ in self.graph[v]:
                if cap > 0 and not visited[to]:
                    visited[to] = True
                    queue.append(to)
        
        return [i for i in range(self.n) if visited[i]]


class FordFulkerson:
    """
    Algorithme de Ford-Fulkerson avec DFS.
    Plus simple mais moins efficace que Dinic.
    """
    
    def __init__(self, n):
        """
        Args:
            n: Nombre de sommets
        """
        self.n = n
        self.graph = [[0] * n for _ in range(n)]
    
    def add_edge(self, u, v, capacity):
        """Ajoute une arête de u vers v"""
        self.graph[u][v] += capacity
    
    def bfs_path(self, source, sink, parent):
        """BFS pour trouver un chemin augmentant"""
        visited = [False] * self.n
        queue = deque([source])
        visited[source] = True
        
        while queue:
            u = queue.popleft()
            
            for v in range(self.n):
                if not visited[v] and self.graph[u][v] > 0:
                    visited[v] = True
                    parent[v] = u
                    queue.append(v)
                    
                    if v == sink:
                        return True
        
        return False
    
    def max_flow(self, source, sink):
        """
        Calcule le flux maximum.
        
        Args:
            source: Sommet source
            sink: Sommet puits
            
        Returns:
            Flux maximum
        """
        parent = [-1] * self.n
        max_flow_value = 0
        
        while self.bfs_path(source, sink, parent):
            # Trouver le flux minimum sur ce chemin
            path_flow = float('inf')
            s = sink
            
            while s != source:
                path_flow = min(path_flow, self.graph[parent[s]][s])
                s = parent[s]
            
            # Mettre à jour les capacités résiduelles
            v = sink
            while v != source:
                u = parent[v]
                self.graph[u][v] -= path_flow
                self.graph[v][u] += path_flow
                v = parent[v]
            
            max_flow_value += path_flow
            parent = [-1] * self.n
        
        return max_flow_value


def bipartite_matching(left_size, right_size, edges):
    """
    Résout le bipartite matching en utilisant max flow.
    
    Args:
        left_size: Taille du groupe gauche
        right_size: Taille du groupe droit
        edges: Liste de tuples (u, v) où u est dans left, v dans right
        
    Returns:
        Nombre maximum de matchings
        
    Example:
        >>> edges = [(0, 0), (0, 1), (1, 1), (2, 0)]
        >>> bipartite_matching(3, 2, edges)
        2
    """
    # Créer un graphe avec source et sink
    # source = left_size + right_size
    # sink = left_size + right_size + 1
    n = left_size + right_size + 2
    source = n - 2
    sink = n - 1
    
    dinic = Dinic(n)
    
    # Source vers tous les sommets gauches
    for i in range(left_size):
        dinic.add_edge(source, i, 1)
    
    # Tous les sommets droits vers sink
    for i in range(right_size):
        dinic.add_edge(left_size + i, sink, 1)
    
    # Arêtes du matching
    for u, v in edges:
        dinic.add_edge(u, left_size + v, 1)
    
    return dinic.max_flow(source, sink)


def test():
    """Tests unitaires complets"""
    
    # Test Dinic basique
    dinic = Dinic(6)
    dinic.add_edge(0, 1, 16)
    dinic.add_edge(0, 2, 13)
    dinic.add_edge(1, 2, 10)
    dinic.add_edge(1, 3, 12)
    dinic.add_edge(2, 1, 4)
    dinic.add_edge(2, 4, 14)
    dinic.add_edge(3, 2, 9)
    dinic.add_edge(3, 5, 20)
    dinic.add_edge(4, 3, 7)
    dinic.add_edge(4, 5, 4)
    
    flow = dinic.max_flow(0, 5)
    assert flow == 23
    
    # Test simple
    dinic2 = Dinic(4)
    dinic2.add_edge(0, 1, 10)
    dinic2.add_edge(0, 2, 10)
    dinic2.add_edge(1, 3, 10)
    dinic2.add_edge(2, 3, 10)
    
    flow2 = dinic2.max_flow(0, 3)
    assert flow2 == 20
    
    # Test Ford-Fulkerson
    ff = FordFulkerson(6)
    ff.add_edge(0, 1, 16)
    ff.add_edge(0, 2, 13)
    ff.add_edge(1, 2, 10)
    ff.add_edge(1, 3, 12)
    ff.add_edge(2, 1, 4)
    ff.add_edge(2, 4, 14)
    ff.add_edge(3, 2, 9)
    ff.add_edge(3, 5, 20)
    ff.add_edge(4, 3, 7)
    ff.add_edge(4, 5, 4)
    
    flow_ff = ff.max_flow(0, 5)
    assert flow_ff == 23
    
    # Test bipartite matching
    edges = [(0, 0), (0, 1), (1, 1), (2, 0)]
    matching = bipartite_matching(3, 2, edges)
    assert matching == 2
    
    # Test bipartite matching complet
    edges2 = [(0, 0), (0, 1), (1, 0), (1, 1)]
    matching2 = bipartite_matching(2, 2, edges2)
    assert matching2 == 2
    
    # Test graphe avec une seule arête
    dinic3 = Dinic(2)
    dinic3.add_edge(0, 1, 5)
    assert dinic3.max_flow(0, 1) == 5
    
    # Test min cut
    dinic4 = Dinic(4)
    dinic4.add_edge(0, 1, 10)
    dinic4.add_edge(0, 2, 10)
    dinic4.add_edge(1, 3, 10)
    dinic4.add_edge(2, 3, 10)
    dinic4.max_flow(0, 3)
    
    cut = dinic4.min_cut(0)
    assert 0 in cut
    assert 3 not in cut
    
    print("Tous les tests passes")


if __name__ == "__main__":
    test()

