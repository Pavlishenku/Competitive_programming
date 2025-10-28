"""
Minimum Spanning Tree (MST) - Kruskal et Prim

Description:
    Algorithmes pour trouver l'arbre couvrant de poids minimum dans
    un graphe pondéré connexe. Kruskal utilise DSU, Prim utilise heap.

Complexité:
    - Kruskal: O(E log E) = O(E log V) avec tri des arêtes
    - Prim: O((V + E) log V) avec heap binaire
    - Espace: O(V + E)

Cas d'usage:
    - Problèmes de connexion minimale (réseau, câbles, routes)
    - Clustering
    - Approximation du TSP
    - Problèmes de minimisation de coût total
    
Problèmes types:
    - Codeforces: 25D, edges, 893C
    - AtCoder: ABC218E, ABC235D
    - CSES: Road Reparation
    
Implémentation par: 2025-10-27
Testé: Oui
"""

import heapq


class DSU:
    """DSU simplifié pour Kruskal"""
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        root_x, root_y = self.find(x), self.find(y)
        if root_x == root_y:
            return False
        if self.rank[root_x] < self.rank[root_y]:
            root_x, root_y = root_y, root_x
        self.parent[root_y] = root_x
        if self.rank[root_x] == self.rank[root_y]:
            self.rank[root_x] += 1
        return True


def kruskal(n, edges):
    """
    Algorithme de Kruskal pour MST.
    
    Args:
        n: Nombre de sommets (0 à n-1)
        edges: Liste de tuples (u, v, poids)
        
    Returns:
        Tuple (poids_total, liste_aretes_mst)
        
    Example:
        >>> edges = [(0, 1, 4), (0, 2, 3), (1, 2, 1), (1, 3, 2), (2, 3, 4)]
        >>> kruskal(4, edges)
        (6, [(1, 2, 1), (1, 3, 2), (0, 2, 3)])
    """
    # Trier les arêtes par poids
    edges = sorted(edges, key=lambda x: x[2])
    
    dsu = DSU(n)
    mst_weight = 0
    mst_edges = []
    
    for u, v, weight in edges:
        if dsu.union(u, v):
            mst_weight += weight
            mst_edges.append((u, v, weight))
            
            # Si on a n-1 arêtes, MST complet
            if len(mst_edges) == n - 1:
                break
    
    # Vérifier si le graphe est connexe
    if len(mst_edges) != n - 1:
        return (float('inf'), [])  # Graphe non connexe
    
    return (mst_weight, mst_edges)


def prim(graph, n):
    """
    Algorithme de Prim pour MST.
    
    Args:
        graph: Dict {sommet: [(voisin, poids), ...]}
        n: Nombre de sommets (0 à n-1)
        
    Returns:
        Tuple (poids_total, liste_aretes_mst)
        
    Example:
        >>> graph = {0: [(1, 4), (2, 3)], 1: [(0, 4), (2, 1), (3, 2)],
        ...          2: [(0, 3), (1, 1), (3, 4)], 3: [(1, 2), (2, 4)]}
        >>> prim(graph, 4)
        (6, [(0, 2), (2, 1), (1, 3)])
    """
    visited = [False] * n
    mst_weight = 0
    mst_edges = []
    
    # Priority queue: (poids, sommet, parent)
    pq = [(0, 0, -1)]  # Commencer du sommet 0
    
    while pq:
        weight, node, parent = heapq.heappop(pq)
        
        if visited[node]:
            continue
        
        visited[node] = True
        mst_weight += weight
        
        if parent != -1:
            mst_edges.append((parent, node))
        
        for neighbor, edge_weight in graph.get(node, []):
            if not visited[neighbor]:
                heapq.heappush(pq, (edge_weight, neighbor, node))
    
    # Vérifier si tous les sommets sont visités
    if len(mst_edges) != n - 1:
        return (float('inf'), [])
    
    return (mst_weight, mst_edges)


def prim_from_start(graph, start):
    """
    Algorithme de Prim à partir d'un sommet spécifique.
    
    Args:
        graph: Dict {sommet: [(voisin, poids), ...]}
        start: Sommet de départ
        
    Returns:
        Tuple (poids_total, liste_aretes_mst, sommets_visites)
    """
    visited = set()
    mst_weight = 0
    mst_edges = []
    
    pq = [(0, start, None)]
    
    while pq:
        weight, node, parent = heapq.heappop(pq)
        
        if node in visited:
            continue
        
        visited.add(node)
        mst_weight += weight
        
        if parent is not None:
            mst_edges.append((parent, node))
        
        for neighbor, edge_weight in graph.get(node, []):
            if neighbor not in visited:
                heapq.heappush(pq, (edge_weight, neighbor, node))
    
    return (mst_weight, mst_edges, visited)


def second_best_mst(n, edges):
    """
    Trouve le deuxième meilleur MST (utile pour certains problèmes).
    
    Args:
        n: Nombre de sommets
        edges: Liste de tuples (u, v, poids)
        
    Returns:
        Poids du second meilleur MST ou float('inf') si n'existe pas
    """
    # Trouver le MST original
    mst_weight, mst_edges = kruskal(n, edges)
    
    if mst_weight == float('inf'):
        return float('inf')
    
    mst_edges_set = set((min(u, v), max(u, v)) for u, v, _ in mst_edges)
    second_best = float('inf')
    
    # Pour chaque arête dans le MST, essayer de la remplacer
    for skip_edge in mst_edges:
        u, v, w = skip_edge
        
        # Essayer de construire un MST sans cette arête
        available_edges = [(a, b, wt) for a, b, wt in edges 
                          if not (min(a, b) == min(u, v) and max(a, b) == max(u, v))]
        
        weight, tree_edges = kruskal(n, available_edges)
        
        if weight != float('inf') and weight > mst_weight:
            second_best = min(second_best, weight)
    
    return second_best


def maximum_spanning_tree(n, edges):
    """
    Trouve l'arbre couvrant de poids maximum (inverse du MST).
    
    Args:
        n: Nombre de sommets
        edges: Liste de tuples (u, v, poids)
        
    Returns:
        Tuple (poids_total, liste_aretes)
    """
    # Inverser les poids et utiliser Kruskal
    inverted_edges = [(u, v, -w) for u, v, w in edges]
    weight, mst_edges = kruskal(n, inverted_edges)
    
    if weight == float('inf'):
        return (float('inf'), [])
    
    # Restaurer les poids originaux
    actual_weight = -weight
    actual_edges = [(u, v, -w) for u, v, w in mst_edges]
    
    return (actual_weight, actual_edges)


def test():
    """Tests unitaires complets"""
    
    # Test graphe exemple
    edges = [
        (0, 1, 4),
        (0, 2, 3),
        (1, 2, 1),
        (1, 3, 2),
        (2, 3, 4)
    ]
    
    # Test Kruskal
    mst_weight, mst_edges = kruskal(4, edges)
    assert mst_weight == 6
    assert len(mst_edges) == 3
    
    # Test Prim
    graph = {
        0: [(1, 4), (2, 3)],
        1: [(0, 4), (2, 1), (3, 2)],
        2: [(0, 3), (1, 1), (3, 4)],
        3: [(1, 2), (2, 4)]
    }
    
    prim_weight, prim_edges = prim(graph, 4)
    assert prim_weight == 6
    assert len(prim_edges) == 3
    
    # Test graphe plus complexe
    edges2 = [
        (0, 1, 10), (0, 2, 6), (0, 3, 5),
        (1, 3, 15), (2, 3, 4)
    ]
    
    weight2, mst2 = kruskal(4, edges2)
    assert weight2 == 19  # 4 + 5 + 10
    
    # Test graphe déconnecté
    edges_disconnected = [
        (0, 1, 1),
        (2, 3, 1)
    ]
    
    weight_disc, mst_disc = kruskal(4, edges_disconnected)
    assert weight_disc == float('inf')
    
    # Test Maximum Spanning Tree
    max_weight, max_edges = maximum_spanning_tree(4, edges)
    assert max_weight == 11  # 4 + 4 + 3
    
    # Test graphe linéaire simple
    linear_edges = [(0, 1, 1), (1, 2, 2), (2, 3, 3)]
    linear_weight, linear_mst = kruskal(4, linear_edges)
    assert linear_weight == 6
    assert len(linear_mst) == 3
    
    # Test avec un seul sommet
    single_weight, single_mst = kruskal(1, [])
    assert single_weight == 0
    assert len(single_mst) == 0
    
    # Test Prim from specific start
    prim_start_weight, prim_start_edges, visited = prim_from_start(graph, 2)
    assert len(visited) == 4
    assert len(prim_start_edges) == 3
    
    print("Tous les tests passes")


if __name__ == "__main__":
    test()

