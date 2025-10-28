"""
Algorithme de Tarjan (SCC, Bridges, Articulation Points)

Description:
    Algorithme de Tarjan pour trouver:
    - Strongly Connected Components (SCC)
    - Bridges (arêtes critiques)
    - Articulation points (sommets critiques)

Complexité:
    - Temps: O(V + E)
    - Espace: O(V)

Cas d'usage:
    - Analyse de connectivité de graphes orientés
    - Trouver les points de défaillance d'un réseau
    - Optimisation de graphes
    - Détection de bi-connectivité
    
Problèmes types:
    - Codeforces: 427C, 652E, 903D
    - AtCoder: ABC245E
    - CSES: Flight Routes Check, Planets Cycles
    
Implémentation par: 2025-10-27
Testé: Oui
"""


class TarjanSCC:
    """
    Algorithme de Tarjan pour Strongly Connected Components.
    """
    
    def __init__(self, graph, n):
        """
        Args:
            graph: Dict {sommet: [voisins]} (graphe orienté)
            n: Nombre de sommets (0 à n-1)
        """
        self.graph = graph
        self.n = n
        self.index = 0
        self.stack = []
        self.on_stack = [False] * n
        self.indices = [-1] * n
        self.lowlinks = [-1] * n
        self.sccs = []
    
    def find_sccs(self):
        """
        Trouve toutes les SCC du graphe.
        
        Returns:
            Liste de SCC, chaque SCC est une liste de sommets
            
        Example:
            >>> graph = {0: [1], 1: [2], 2: [0], 3: [1, 4], 4: [5], 5: [3]}
            >>> tarjan = TarjanSCC(graph, 6)
            >>> tarjan.find_sccs()
            [[0, 2, 1], [3, 5, 4]]
        """
        for v in range(self.n):
            if self.indices[v] == -1:
                self._strongconnect(v)
        
        return self.sccs
    
    def _strongconnect(self, v):
        """DFS modifié pour trouver les SCC"""
        self.indices[v] = self.index
        self.lowlinks[v] = self.index
        self.index += 1
        self.stack.append(v)
        self.on_stack[v] = True
        
        for w in self.graph.get(v, []):
            if self.indices[w] == -1:
                self._strongconnect(w)
                self.lowlinks[v] = min(self.lowlinks[v], self.lowlinks[w])
            elif self.on_stack[w]:
                self.lowlinks[v] = min(self.lowlinks[v], self.indices[w])
        
        # Si v est une racine de SCC
        if self.lowlinks[v] == self.indices[v]:
            scc = []
            while True:
                w = self.stack.pop()
                self.on_stack[w] = False
                scc.append(w)
                if w == v:
                    break
            self.sccs.append(scc)


class TarjanBridges:
    """
    Trouver les bridges (arêtes critiques) dans un graphe non-orienté.
    """
    
    def __init__(self, graph, n):
        """
        Args:
            graph: Dict {sommet: [voisins]} (graphe non-orienté)
            n: Nombre de sommets
        """
        self.graph = graph
        self.n = n
        self.time = 0
        self.visited = [False] * n
        self.disc = [0] * n
        self.low = [0] * n
        self.bridges = []
    
    def find_bridges(self):
        """
        Trouve toutes les bridges.
        
        Returns:
            Liste de tuples (u, v) représentant les bridges
            
        Example:
            >>> graph = {0: [1, 2], 1: [0, 2], 2: [0, 1, 3], 3: [2]}
            >>> tb = TarjanBridges(graph, 4)
            >>> tb.find_bridges()
            [(2, 3)]
        """
        for i in range(self.n):
            if not self.visited[i]:
                self._dfs_bridge(i, -1)
        
        return self.bridges
    
    def _dfs_bridge(self, u, parent):
        """DFS pour trouver les bridges"""
        self.visited[u] = True
        self.disc[u] = self.low[u] = self.time
        self.time += 1
        
        for v in self.graph.get(u, []):
            if not self.visited[v]:
                self._dfs_bridge(v, u)
                self.low[u] = min(self.low[u], self.low[v])
                
                # Si low[v] > disc[u], alors (u, v) est une bridge
                if self.low[v] > self.disc[u]:
                    self.bridges.append((u, v))
            
            elif v != parent:
                self.low[u] = min(self.low[u], self.disc[v])


class TarjanArticulationPoints:
    """
    Trouver les articulation points (cut vertices) dans un graphe non-orienté.
    """
    
    def __init__(self, graph, n):
        """
        Args:
            graph: Dict {sommet: [voisins]}
            n: Nombre de sommets
        """
        self.graph = graph
        self.n = n
        self.time = 0
        self.visited = [False] * n
        self.disc = [0] * n
        self.low = [0] * n
        self.parent = [-1] * n
        self.articulation_points = set()
    
    def find_articulation_points(self):
        """
        Trouve tous les articulation points.
        
        Returns:
            Liste des articulation points
            
        Example:
            >>> graph = {0: [1], 1: [0, 2], 2: [1, 3, 4], 3: [2], 4: [2]}
            >>> tap = TarjanArticulationPoints(graph, 5)
            >>> sorted(tap.find_articulation_points())
            [1, 2]
        """
        for i in range(self.n):
            if not self.visited[i]:
                self._dfs_ap(i)
        
        return list(self.articulation_points)
    
    def _dfs_ap(self, u):
        """DFS pour trouver les articulation points"""
        children = 0
        self.visited[u] = True
        self.disc[u] = self.low[u] = self.time
        self.time += 1
        
        for v in self.graph.get(u, []):
            if not self.visited[v]:
                children += 1
                self.parent[v] = u
                self._dfs_ap(v)
                
                self.low[u] = min(self.low[u], self.low[v])
                
                # u est un articulation point si:
                # (1) u est racine avec 2+ enfants
                # (2) u n'est pas racine et low[v] >= disc[u]
                if self.parent[u] == -1 and children > 1:
                    self.articulation_points.add(u)
                
                if self.parent[u] != -1 and self.low[v] >= self.disc[u]:
                    self.articulation_points.add(u)
            
            elif v != self.parent[u]:
                self.low[u] = min(self.low[u], self.disc[v])


def kosaraju_scc(graph, n):
    """
    Alternative à Tarjan: algorithme de Kosaraju pour SCC.
    Plus simple mais nécessite deux DFS.
    
    Args:
        graph: Dict {sommet: [voisins]} (graphe orienté)
        n: Nombre de sommets
        
    Returns:
        Liste de SCC
    """
    # Premier DFS pour ordre de finition
    visited = [False] * n
    stack = []
    
    def dfs1(v):
        visited[v] = True
        for u in graph.get(v, []):
            if not visited[u]:
                dfs1(u)
        stack.append(v)
    
    for i in range(n):
        if not visited[i]:
            dfs1(i)
    
    # Créer le graphe transposé
    transpose = {i: [] for i in range(n)}
    for u in graph:
        for v in graph[u]:
            transpose[v].append(u)
    
    # Deuxième DFS sur graphe transposé
    visited = [False] * n
    sccs = []
    
    def dfs2(v, scc):
        visited[v] = True
        scc.append(v)
        for u in transpose.get(v, []):
            if not visited[u]:
                dfs2(u, scc)
    
    while stack:
        v = stack.pop()
        if not visited[v]:
            scc = []
            dfs2(v, scc)
            sccs.append(scc)
    
    return sccs


def test():
    """Tests unitaires complets"""
    
    # Test SCC
    graph_scc = {
        0: [1],
        1: [2],
        2: [0],
        3: [1, 4],
        4: [5],
        5: [3]
    }
    
    tarjan = TarjanSCC(graph_scc, 6)
    sccs = tarjan.find_sccs()
    assert len(sccs) == 2
    
    # Test Kosaraju
    sccs_kosaraju = kosaraju_scc(graph_scc, 6)
    assert len(sccs_kosaraju) == 2
    
    # Test Bridges
    graph_bridge = {
        0: [1, 2],
        1: [0, 2],
        2: [0, 1, 3],
        3: [2]
    }
    
    tb = TarjanBridges(graph_bridge, 4)
    bridges = tb.find_bridges()
    assert len(bridges) == 1
    assert (2, 3) in bridges or (3, 2) in bridges
    
    # Test Articulation Points
    graph_ap = {
        0: [1],
        1: [0, 2],
        2: [1, 3, 4],
        3: [2],
        4: [2]
    }
    
    tap = TarjanArticulationPoints(graph_ap, 5)
    aps = tap.find_articulation_points()
    assert 1 in aps
    assert 2 in aps
    assert len(aps) == 2
    
    # Test graphe simple sans bridge
    graph_no_bridge = {
        0: [1, 2],
        1: [0, 2],
        2: [0, 1]
    }
    
    tb2 = TarjanBridges(graph_no_bridge, 3)
    bridges2 = tb2.find_bridges()
    assert len(bridges2) == 0
    
    # Test graphe linéaire (tous bridges)
    graph_linear = {
        0: [1],
        1: [0, 2],
        2: [1, 3],
        3: [2]
    }
    
    tb3 = TarjanBridges(graph_linear, 4)
    bridges3 = tb3.find_bridges()
    assert len(bridges3) == 3
    
    # Test SCC sur graphe fortement connexe
    graph_strong = {
        0: [1],
        1: [2],
        2: [0]
    }
    
    tarjan2 = TarjanSCC(graph_strong, 3)
    sccs2 = tarjan2.find_sccs()
    assert len(sccs2) == 1
    assert len(sccs2[0]) == 3
    
    print("Tous les tests passes")


if __name__ == "__main__":
    test()

