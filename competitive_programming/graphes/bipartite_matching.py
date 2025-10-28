"""
Bipartite Matching (Couplage Biparti)

Description:
    Algorithmes pour trouver le couplage maximum dans un graphe biparti.
    Implémentations: Kuhn (simple), Hopcroft-Karp (optimal).

Complexité:
    - Kuhn (Hungarian): O(VE)
    - Hopcroft-Karp: O(E√V)

Cas d'usage:
    - Affectation de tâches
    - Problèmes d'appariement
    - Couverture de sommets minimum
    
Problèmes types:
    - Codeforces: 277E, 277A
    - SPOJ: MATCHING
    
Implémentation par: 2025-10-27
Testé: Oui
"""

from collections import deque


class KuhnMatching:
    """
    Algorithme de Kuhn pour bipartite matching.
    Simple et efficace pour graphes de taille moyenne.
    """
    
    def __init__(self, n_left, n_right):
        """
        Args:
            n_left: Nombre de sommets à gauche
            n_right: Nombre de sommets à droite
        """
        self.n_left = n_left
        self.n_right = n_right
        self.graph = [[] for _ in range(n_left)]
        self.match_right = [-1] * n_right
        self.match_left = [-1] * n_left
    
    def add_edge(self, u, v):
        """
        Ajoute une arête de gauche u vers droite v.
        
        Args:
            u: Sommet gauche (0-indexed)
            v: Sommet droit (0-indexed)
        """
        self.graph[u].append(v)
    
    def _dfs(self, u, visited):
        """DFS pour trouver un chemin augmentant"""
        for v in self.graph[u]:
            if visited[v]:
                continue
            
            visited[v] = True
            
            if self.match_right[v] == -1 or self._dfs(self.match_right[v], visited):
                self.match_right[v] = u
                self.match_left[u] = v
                return True
        
        return False
    
    def max_matching(self):
        """
        Trouve le couplage maximum.
        
        Returns:
            Taille du couplage maximum
            
        Example:
            >>> km = KuhnMatching(3, 3)
            >>> km.add_edge(0, 0)
            >>> km.add_edge(0, 1)
            >>> km.add_edge(1, 1)
            >>> km.add_edge(2, 2)
            >>> km.max_matching()
            3
        """
        matching = 0
        
        for u in range(self.n_left):
            visited = [False] * self.n_right
            if self._dfs(u, visited):
                matching += 1
        
        return matching
    
    def get_matching_pairs(self):
        """
        Retourne les paires du couplage.
        
        Returns:
            Liste de tuples (left, right)
        """
        pairs = []
        for u in range(self.n_left):
            if self.match_left[u] != -1:
                pairs.append((u, self.match_left[u]))
        return pairs


class HopcroftKarp:
    """
    Algorithme de Hopcroft-Karp pour bipartite matching.
    Plus rapide que Kuhn: O(E√V).
    """
    
    def __init__(self, n_left, n_right):
        """
        Args:
            n_left: Nombre de sommets à gauche
            n_right: Nombre de sommets à droite
        """
        self.n_left = n_left
        self.n_right = n_right
        self.graph = [[] for _ in range(n_left)]
        self.pair_u = [-1] * n_left
        self.pair_v = [-1] * n_right
        self.dist = [0] * (n_left + 1)
        self.INF = float('inf')
        self.NIL = n_left  # Noeud NIL
    
    def add_edge(self, u, v):
        """Ajoute une arête"""
        self.graph[u].append(v)
    
    def _bfs(self):
        """BFS pour construire les niveaux"""
        queue = deque()
        
        for u in range(self.n_left):
            if self.pair_u[u] == -1:
                self.dist[u] = 0
                queue.append(u)
            else:
                self.dist[u] = self.INF
        
        self.dist[self.NIL] = self.INF
        
        while queue:
            u = queue.popleft()
            
            if self.dist[u] < self.dist[self.NIL]:
                for v in self.graph[u]:
                    if self.dist[self.pair_v[v]] == self.INF:
                        self.dist[self.pair_v[v]] = self.dist[u] + 1
                        if self.pair_v[v] != -1:
                            queue.append(self.pair_v[v])
                        else:
                            self.dist[self.NIL] = self.dist[u] + 1
        
        return self.dist[self.NIL] != self.INF
    
    def _dfs(self, u):
        """DFS pour trouver des chemins augmentants"""
        if u == self.NIL:
            return True
        
        for v in self.graph[u]:
            next_u = self.pair_v[v] if self.pair_v[v] != -1 else self.NIL
            
            if self.dist[next_u] == self.dist[u] + 1:
                if self._dfs(next_u):
                    self.pair_v[v] = u
                    self.pair_u[u] = v
                    return True
        
        self.dist[u] = self.INF
        return False
    
    def max_matching(self):
        """
        Trouve le couplage maximum.
        
        Returns:
            Taille du couplage maximum
        """
        matching = 0
        
        while self._bfs():
            for u in range(self.n_left):
                if self.pair_u[u] == -1:
                    if self._dfs(u):
                        matching += 1
        
        return matching
    
    def get_matching_pairs(self):
        """Retourne les paires du couplage"""
        pairs = []
        for u in range(self.n_left):
            if self.pair_u[u] != -1:
                pairs.append((u, self.pair_u[u]))
        return pairs


def minimum_vertex_cover(n_left, n_right, edges):
    """
    Trouve la couverture de sommets minimum dans un graphe biparti.
    Par le théorème de König: min vertex cover = max matching.
    
    Args:
        n_left: Nombre de sommets gauche
        n_right: Nombre de sommets droit
        edges: Liste de tuples (u, v)
        
    Returns:
        Taille de la couverture minimum
    """
    km = KuhnMatching(n_left, n_right)
    for u, v in edges:
        km.add_edge(u, v)
    
    return km.max_matching()


def maximum_independent_set(n_left, n_right, edges):
    """
    Trouve l'ensemble indépendant maximum dans un graphe biparti.
    Max independent set = n - max matching.
    
    Args:
        n_left: Nombre de sommets gauche
        n_right: Nombre de sommets droit
        edges: Liste de tuples (u, v)
        
    Returns:
        Taille de l'ensemble indépendant maximum
    """
    matching = minimum_vertex_cover(n_left, n_right, edges)
    return n_left + n_right - matching


def test():
    """Tests unitaires complets"""
    
    # Test Kuhn
    km = KuhnMatching(3, 3)
    km.add_edge(0, 0)
    km.add_edge(0, 1)
    km.add_edge(1, 1)
    km.add_edge(1, 2)
    km.add_edge(2, 2)
    
    matching = km.max_matching()
    assert matching == 3
    
    pairs = km.get_matching_pairs()
    assert len(pairs) == 3
    
    # Test Hopcroft-Karp
    hk = HopcroftKarp(3, 3)
    hk.add_edge(0, 0)
    hk.add_edge(0, 1)
    hk.add_edge(1, 1)
    hk.add_edge(1, 2)
    hk.add_edge(2, 2)
    
    matching_hk = hk.max_matching()
    assert matching_hk == 3
    
    # Test avec matching partiel
    km2 = KuhnMatching(3, 2)
    km2.add_edge(0, 0)
    km2.add_edge(1, 0)
    km2.add_edge(2, 1)
    
    matching2 = km2.max_matching()
    assert matching2 == 2
    
    # Test graphe complet biparti
    km3 = KuhnMatching(2, 2)
    km3.add_edge(0, 0)
    km3.add_edge(0, 1)
    km3.add_edge(1, 0)
    km3.add_edge(1, 1)
    
    matching3 = km3.max_matching()
    assert matching3 == 2
    
    # Test minimum vertex cover
    edges = [(0, 0), (0, 1), (1, 1), (2, 2)]
    cover = minimum_vertex_cover(3, 3, edges)
    assert cover == 3
    
    # Test maximum independent set
    indep = maximum_independent_set(3, 3, edges)
    assert indep == 3  # 6 sommets - 3 matching
    
    # Test graphe sans arêtes
    km_empty = KuhnMatching(3, 3)
    assert km_empty.max_matching() == 0
    
    # Test Hopcroft-Karp sur grand graphe
    hk_large = HopcroftKarp(100, 100)
    for i in range(100):
        hk_large.add_edge(i, i)
    
    assert hk_large.max_matching() == 100
    
    print("Tous les tests passes")


if __name__ == "__main__":
    test()

