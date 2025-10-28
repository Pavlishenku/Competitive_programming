"""
Algorithmes de Plus Courts Chemins (Bellman-Ford, Floyd-Warshall)

Description:
    Bellman-Ford: Plus courts chemins depuis une source, supporte poids négatifs
    Floyd-Warshall: Plus courts chemins entre toutes paires de sommets

Complexité:
    - Bellman-Ford: O(VE) temps, O(V) espace
    - Floyd-Warshall: O(V³) temps, O(V²) espace

Cas d'usage:
    - Graphes avec poids négatifs
    - Détection de cycles négatifs
    - Plus courts chemins entre toutes paires
    - Problèmes de transitivité
    
Problèmes types:
    - Codeforces: 295B, 464E, 567E
    - AtCoder: ABC137E, ABC143E
    - CSES: High Score, Shortest Routes II
    
Implémentation par: 2025-10-27
Testé: Oui
"""

from collections import defaultdict


def bellman_ford(graph, start, n):
    """
    Algorithme de Bellman-Ford pour plus courts chemins avec poids négatifs.
    
    Args:
        graph: Liste d'arêtes [(u, v, poids), ...]
        start: Sommet de départ
        n: Nombre de sommets (0 à n-1)
        
    Returns:
        Tuple (distances, has_negative_cycle)
        distances: Dict {sommet: distance} ou None si cycle négatif
        
    Example:
        >>> edges = [(0, 1, 4), (0, 2, 5), (1, 2, -3), (2, 3, 4)]
        >>> bellman_ford(edges, 0, 4)
        ({0: 0, 1: 4, 2: 1, 3: 5}, False)
    """
    distances = defaultdict(lambda: float('inf'))
    distances[start] = 0
    
    # Relaxation V-1 fois
    for _ in range(n - 1):
        updated = False
        for u, v, weight in graph:
            if distances[u] != float('inf') and distances[u] + weight < distances[v]:
                distances[v] = distances[u] + weight
                updated = True
        
        # Optimisation: si aucune mise à jour, on peut arrêter
        if not updated:
            break
    
    # Vérifier les cycles négatifs
    has_negative_cycle = False
    for u, v, weight in graph:
        if distances[u] != float('inf') and distances[u] + weight < distances[v]:
            has_negative_cycle = True
            break
    
    if has_negative_cycle:
        return (None, True)
    
    return (dict(distances), False)


def bellman_ford_with_path(graph, start, end, n):
    """
    Bellman-Ford qui retourne aussi le chemin.
    
    Args:
        graph: Liste d'arêtes [(u, v, poids), ...]
        start: Sommet de départ
        end: Sommet d'arrivée
        n: Nombre de sommets
        
    Returns:
        Tuple (distance, chemin) ou (None, None) si cycle négatif
    """
    distances = defaultdict(lambda: float('inf'))
    distances[start] = 0
    parent = {}
    
    for _ in range(n - 1):
        for u, v, weight in graph:
            if distances[u] != float('inf') and distances[u] + weight < distances[v]:
                distances[v] = distances[u] + weight
                parent[v] = u
    
    # Vérifier cycle négatif
    for u, v, weight in graph:
        if distances[u] != float('inf') and distances[u] + weight < distances[v]:
            return (None, None)
    
    # Reconstruire le chemin
    if end not in parent and end != start:
        return (float('inf'), [])
    
    path = []
    current = end
    while current != start:
        path.append(current)
        if current not in parent:
            return (float('inf'), [])
        current = parent[current]
    path.append(start)
    path.reverse()
    
    return (distances[end], path)


def detect_negative_cycle(graph, n):
    """
    Détecte et retourne un cycle négatif s'il existe.
    
    Args:
        graph: Liste d'arêtes [(u, v, poids), ...]
        n: Nombre de sommets
        
    Returns:
        Liste des sommets du cycle ou [] si pas de cycle négatif
    """
    distances = [0] * n  # Initialiser à 0 pour détecter tous les cycles
    parent = [-1] * n
    last_updated = -1
    
    # Relaxation n fois pour trouver le cycle
    for i in range(n):
        last_updated = -1
        for u, v, weight in graph:
            if distances[u] + weight < distances[v]:
                distances[v] = distances[u] + weight
                parent[v] = u
                last_updated = v
    
    if last_updated == -1:
        return []  # Pas de cycle négatif
    
    # Remonter pour trouver un sommet du cycle
    cycle_node = last_updated
    for _ in range(n):
        cycle_node = parent[cycle_node]
    
    # Extraire le cycle
    cycle = []
    current = cycle_node
    while True:
        cycle.append(current)
        current = parent[current]
        if current == cycle_node:
            break
    
    cycle.reverse()
    return cycle


def floyd_warshall(graph, n):
    """
    Algorithme de Floyd-Warshall pour tous les plus courts chemins.
    
    Args:
        graph: Dict {u: {v: poids}} ou matrice d'adjacence
        n: Nombre de sommets (0 à n-1)
        
    Returns:
        Matrice de distances [n][n]
        
    Example:
        >>> graph = {0: {1: 3, 2: 8}, 1: {3: 1}, 2: {1: 4}, 3: {}}
        >>> dist = floyd_warshall(graph, 4)
        >>> dist[0][3]
        4
    """
    # Initialiser la matrice de distances
    dist = [[float('inf')] * n for _ in range(n)]
    
    # Distance de chaque sommet à lui-même est 0
    for i in range(n):
        dist[i][i] = 0
    
    # Remplir avec les arêtes existantes
    if isinstance(graph, dict):
        for u in graph:
            for v, weight in graph[u].items():
                dist[u][v] = weight
    else:  # Liste d'arêtes
        for u, v, weight in graph:
            dist[u][v] = min(dist[u][v], weight)
    
    # Floyd-Warshall
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i][k] != float('inf') and dist[k][j] != float('inf'):
                    dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
    
    return dist


def floyd_warshall_with_path(graph, n):
    """
    Floyd-Warshall qui permet de reconstruire les chemins.
    
    Args:
        graph: Dict ou liste d'arêtes
        n: Nombre de sommets
        
    Returns:
        Tuple (matrice_distances, matrice_next)
        matrice_next permet de reconstruire les chemins
    """
    dist = [[float('inf')] * n for _ in range(n)]
    next_node = [[None] * n for _ in range(n)]
    
    for i in range(n):
        dist[i][i] = 0
        next_node[i][i] = i
    
    if isinstance(graph, dict):
        for u in graph:
            for v, weight in graph[u].items():
                dist[u][v] = weight
                next_node[u][v] = v
    else:
        for u, v, weight in graph:
            if weight < dist[u][v]:
                dist[u][v] = weight
                next_node[u][v] = v
    
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i][k] != float('inf') and dist[k][j] != float('inf'):
                    if dist[i][k] + dist[k][j] < dist[i][j]:
                        dist[i][j] = dist[i][k] + dist[k][j]
                        next_node[i][j] = next_node[i][k]
    
    return (dist, next_node)


def reconstruct_path(next_matrix, start, end):
    """
    Reconstruit le chemin depuis la matrice next de Floyd-Warshall.
    
    Args:
        next_matrix: Matrice retournée par floyd_warshall_with_path
        start: Sommet de départ
        end: Sommet d'arrivée
        
    Returns:
        Liste des sommets du chemin
    """
    if next_matrix[start][end] is None:
        return []
    
    path = [start]
    current = start
    
    while current != end:
        current = next_matrix[current][end]
        path.append(current)
    
    return path


def test():
    """Tests unitaires complets"""
    
    # Test Bellman-Ford
    edges = [
        (0, 1, 4),
        (0, 2, 5),
        (1, 2, -3),
        (2, 3, 4)
    ]
    
    distances, has_cycle = bellman_ford(edges, 0, 4)
    assert not has_cycle
    assert distances[0] == 0
    assert distances[1] == 4
    assert distances[2] == 1
    assert distances[3] == 5
    
    # Test Bellman-Ford avec chemin
    dist, path = bellman_ford_with_path(edges, 0, 3, 4)
    assert dist == 5
    assert path[0] == 0 and path[-1] == 3
    
    # Test cycle négatif
    edges_negative = [
        (0, 1, 1),
        (1, 2, -3),
        (2, 0, 1)
    ]
    
    distances_neg, has_cycle_neg = bellman_ford(edges_negative, 0, 3)
    assert has_cycle_neg
    assert distances_neg is None
    
    cycle = detect_negative_cycle(edges_negative, 3)
    assert len(cycle) > 0
    
    # Test Floyd-Warshall
    graph = {
        0: {1: 3, 2: 8},
        1: {3: 1},
        2: {1: 4},
        3: {}
    }
    
    dist = floyd_warshall(graph, 4)
    assert dist[0][0] == 0
    assert dist[0][1] == 3
    assert dist[0][3] == 4
    assert dist[2][3] == 5
    
    # Test Floyd-Warshall avec reconstruction de chemin
    dist_with_path, next_matrix = floyd_warshall_with_path(graph, 4)
    path_fw = reconstruct_path(next_matrix, 0, 3)
    assert path_fw[0] == 0 and path_fw[-1] == 3
    
    # Test graphe avec poids négatifs (pas de cycle)
    edges_neg_no_cycle = [
        (0, 1, 5),
        (1, 2, -3),
        (2, 3, 2)
    ]
    
    dist_neg_no, has_cycle_no = bellman_ford(edges_neg_no_cycle, 0, 4)
    assert not has_cycle_no
    assert dist_neg_no[3] == 4  # 5 + (-3) + 2
    
    # Test Floyd-Warshall sur graphe complet
    complete_edges = [
        (0, 1, 1), (0, 2, 4),
        (1, 0, 1), (1, 2, 2),
        (2, 0, 4), (2, 1, 2)
    ]
    
    dist_complete = floyd_warshall(complete_edges, 3)
    assert dist_complete[0][2] == 3  # Via 1: 1 + 2
    
    print("Tous les tests passes")


if __name__ == "__main__":
    test()

