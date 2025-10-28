"""
DFS et BFS (Parcours de Graphes)

Description:
    Algorithmes fondamentaux de parcours de graphes.
    DFS (Depth-First Search) explore en profondeur.
    BFS (Breadth-First Search) explore par niveaux.

Complexité:
    - Temps: O(V + E) où V = sommets, E = arêtes
    - Espace: O(V) pour visited array

Cas d'usage:
    - Détection de composantes connexes
    - Détection de cycles
    - Plus court chemin (BFS sur graphe non pondéré)
    - Tri topologique (DFS)
    - Pathfinding basique
    
Problèmes types:
    - Codeforces: 115A, 580C, 377A
    - AtCoder: ABC138D, ABC160D
    - CSES: Building Roads, Message Route
    
Implémentation par: 2025-10-27
Testé: Oui
"""

from collections import deque, defaultdict


def dfs(graph, start, visited=None):
    """
    DFS récursif sur un graphe.
    
    Args:
        graph: Dictionnaire {sommet: [voisins]} ou liste d'adjacence
        start: Sommet de départ
        visited: Set des sommets visités (optionnel)
        
    Returns:
        Liste des sommets visités dans l'ordre DFS
        
    Example:
        >>> graph = {0: [1, 2], 1: [0, 3], 2: [0], 3: [1]}
        >>> dfs(graph, 0)
        [0, 1, 3, 2]
    """
    if visited is None:
        visited = set()
    
    result = []
    
    def _dfs(node):
        visited.add(node)
        result.append(node)
        
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                _dfs(neighbor)
    
    _dfs(start)
    return result


def dfs_iterative(graph, start):
    """
    DFS itératif (évite récursion profonde).
    Recommandé pour competitive programming.
    
    Args:
        graph: Dictionnaire {sommet: [voisins]} ou liste d'adjacence
        start: Sommet de départ
        
    Returns:
        Liste des sommets visités dans l'ordre DFS
    """
    visited = set()
    stack = [start]
    result = []
    
    while stack:
        node = stack.pop()
        if node not in visited:
            visited.add(node)
            result.append(node)
            
            # Ajouter les voisins dans l'ordre inverse pour garder l'ordre
            for neighbor in reversed(graph.get(node, [])):
                if neighbor not in visited:
                    stack.append(neighbor)
    
    return result


def bfs(graph, start):
    """
    BFS sur un graphe.
    
    Args:
        graph: Dictionnaire {sommet: [voisins]} ou liste d'adjacence
        start: Sommet de départ
        
    Returns:
        Liste des sommets visités dans l'ordre BFS
        
    Example:
        >>> graph = {0: [1, 2], 1: [0, 3], 2: [0], 3: [1]}
        >>> bfs(graph, 0)
        [0, 1, 2, 3]
    """
    visited = set([start])
    queue = deque([start])
    result = []
    
    while queue:
        node = queue.popleft()
        result.append(node)
        
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    
    return result


def bfs_shortest_path(graph, start, end):
    """
    BFS pour trouver le plus court chemin (graphe non pondéré).
    
    Args:
        graph: Dictionnaire {sommet: [voisins]}
        start: Sommet de départ
        end: Sommet d'arrivée
        
    Returns:
        Tuple (distance, chemin) ou (float('inf'), []) si pas de chemin
    """
    if start == end:
        return (0, [start])
    
    visited = {start}
    queue = deque([(start, [start])])
    
    while queue:
        node, path = queue.popleft()
        
        for neighbor in graph.get(node, []):
            if neighbor == end:
                return (len(path), path + [neighbor])
            
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
    
    return (float('inf'), [])


def find_all_components(graph):
    """
    Trouve toutes les composantes connexes d'un graphe.
    
    Args:
        graph: Dictionnaire {sommet: [voisins]}
        
    Returns:
        Liste de composantes, chaque composante est une liste de sommets
        
    Example:
        >>> graph = {0: [1], 1: [0], 2: [3], 3: [2], 4: []}
        >>> find_all_components(graph)
        [[0, 1], [2, 3], [4]]
    """
    visited = set()
    components = []
    
    for node in graph:
        if node not in visited:
            component = dfs_iterative({k: v for k, v in graph.items()}, node)
            visited.update(component)
            components.append(component)
    
    return components


def has_cycle_undirected(graph, n):
    """
    Détecte si un graphe non-orienté contient un cycle.
    
    Args:
        graph: Dictionnaire {sommet: [voisins]}
        n: Nombre de sommets (0 à n-1)
        
    Returns:
        True si cycle détecté, False sinon
    """
    visited = set()
    
    def dfs_cycle(node, parent):
        visited.add(node)
        
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                if dfs_cycle(neighbor, node):
                    return True
            elif neighbor != parent:
                return True
        
        return False
    
    for i in range(n):
        if i not in visited:
            if dfs_cycle(i, -1):
                return True
    
    return False


def has_cycle_directed(graph, n):
    """
    Détecte si un graphe orienté contient un cycle.
    
    Args:
        graph: Dictionnaire {sommet: [voisins]}
        n: Nombre de sommets (0 à n-1)
        
    Returns:
        True si cycle détecté, False sinon
    """
    WHITE, GRAY, BLACK = 0, 1, 2
    color = [WHITE] * n
    
    def dfs_cycle(node):
        if color[node] == GRAY:
            return True
        if color[node] == BLACK:
            return False
        
        color[node] = GRAY
        
        for neighbor in graph.get(node, []):
            if dfs_cycle(neighbor):
                return True
        
        color[node] = BLACK
        return False
    
    for i in range(n):
        if color[i] == WHITE:
            if dfs_cycle(i):
                return True
    
    return False


def topological_sort(graph, n):
    """
    Tri topologique d'un DAG (Directed Acyclic Graph).
    
    Args:
        graph: Dictionnaire {sommet: [voisins]}
        n: Nombre de sommets (0 à n-1)
        
    Returns:
        Liste des sommets en ordre topologique, ou [] si cycle
    """
    visited = [False] * n
    stack = []
    
    def dfs_topo(node):
        visited[node] = True
        
        for neighbor in graph.get(node, []):
            if not visited[neighbor]:
                dfs_topo(neighbor)
        
        stack.append(node)
    
    # Vérifier d'abord s'il y a un cycle
    if has_cycle_directed(graph, n):
        return []
    
    for i in range(n):
        if not visited[i]:
            dfs_topo(i)
    
    return stack[::-1]


def bfs_distance(graph, start):
    """
    Calcule la distance de start à tous les autres sommets.
    
    Args:
        graph: Dictionnaire {sommet: [voisins]}
        start: Sommet de départ
        
    Returns:
        Dictionnaire {sommet: distance}
    """
    distances = {start: 0}
    queue = deque([start])
    
    while queue:
        node = queue.popleft()
        current_dist = distances[node]
        
        for neighbor in graph.get(node, []):
            if neighbor not in distances:
                distances[neighbor] = current_dist + 1
                queue.append(neighbor)
    
    return distances


def test():
    """Tests unitaires complets"""
    
    # Test graphe simple
    graph = {
        0: [1, 2],
        1: [0, 3, 4],
        2: [0, 5],
        3: [1],
        4: [1],
        5: [2]
    }
    
    # Test DFS
    dfs_result = dfs(graph, 0)
    assert len(dfs_result) == 6
    assert dfs_result[0] == 0
    
    # Test DFS iteratif
    dfs_iter_result = dfs_iterative(graph, 0)
    assert len(dfs_iter_result) == 6
    assert dfs_iter_result[0] == 0
    
    # Test BFS
    bfs_result = bfs(graph, 0)
    assert bfs_result == [0, 1, 2, 3, 4, 5]
    
    # Test plus court chemin
    dist, path = bfs_shortest_path(graph, 0, 4)
    assert dist == 2
    assert path[0] == 0 and path[-1] == 4
    
    # Test composantes connexes
    graph2 = {
        0: [1], 1: [0],
        2: [3], 3: [2],
        4: []
    }
    components = find_all_components(graph2)
    assert len(components) == 3
    
    # Test détection de cycle (non-orienté)
    graph_no_cycle = {0: [1, 2], 1: [0], 2: [0]}
    assert not has_cycle_undirected(graph_no_cycle, 3)
    
    graph_with_cycle = {0: [1], 1: [0, 2], 2: [1, 0]}
    assert has_cycle_undirected(graph_with_cycle, 3)
    
    # Test détection de cycle (orienté)
    graph_dag = {0: [1, 2], 1: [3], 2: [3], 3: []}
    assert not has_cycle_directed(graph_dag, 4)
    
    graph_cycle = {0: [1], 1: [2], 2: [0]}
    assert has_cycle_directed(graph_cycle, 3)
    
    # Test tri topologique
    topo = topological_sort(graph_dag, 4)
    assert len(topo) == 4
    assert topo[0] == 0
    assert topo[-1] == 3
    
    # Test BFS distances
    distances = bfs_distance(graph, 0)
    assert distances[0] == 0
    assert distances[1] == 1
    assert distances[3] == 2
    
    print("Tous les tests passes")


if __name__ == "__main__":
    test()

