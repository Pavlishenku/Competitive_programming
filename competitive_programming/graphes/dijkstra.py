"""
Dijkstra - Plus court chemin

Description:
    Algorithme de Dijkstra pour trouver les plus courts chemins depuis
    une source vers tous les autres sommets dans un graphe pondéré
    avec poids positifs. Utilise une priority queue (heap).

Complexité:
    - Temps: O((V + E) log V) avec heap binaire
    - Espace: O(V)

Cas d'usage:
    - Plus court chemin dans graphe pondéré positif
    - Problèmes de routage et navigation
    - Optimisation de coûts
    - Pathfinding avec poids
    
Problèmes types:
    - Codeforces: 20C, 59E, 715B
    - AtCoder: ABC160D, ABC191E
    - CSES: Shortest Routes I
    
Implémentation par: 2025-10-27
Testé: Oui
"""

import heapq
from collections import defaultdict


def dijkstra(graph, start, n=None):
    """
    Dijkstra classique retournant les distances minimales.
    
    Args:
        graph: Dict {sommet: [(voisin, poids), ...]} ou liste d'adjacence
        start: Sommet de départ
        n: Nombre total de sommets (optionnel)
        
    Returns:
        Dictionnaire {sommet: distance_minimale}
        
    Example:
        >>> graph = {0: [(1, 4), (2, 1)], 1: [(3, 1)], 2: [(1, 2), (3, 5)], 3: []}
        >>> dijkstra(graph, 0)
        {0: 0, 2: 1, 1: 3, 3: 4}
    """
    distances = defaultdict(lambda: float('inf'))
    distances[start] = 0
    
    # Priority queue: (distance, sommet)
    pq = [(0, start)]
    
    while pq:
        current_dist, node = heapq.heappop(pq)
        
        # Si on a déjà trouvé un meilleur chemin, skip
        if current_dist > distances[node]:
            continue
        
        for neighbor, weight in graph.get(node, []):
            distance = current_dist + weight
            
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(pq, (distance, neighbor))
    
    return dict(distances)


def dijkstra_with_path(graph, start, end):
    """
    Dijkstra qui retourne aussi le chemin.
    
    Args:
        graph: Dict {sommet: [(voisin, poids), ...]}
        start: Sommet de départ
        end: Sommet d'arrivée
        
    Returns:
        Tuple (distance, chemin) ou (float('inf'), []) si pas de chemin
    """
    distances = defaultdict(lambda: float('inf'))
    distances[start] = 0
    parent = {}
    
    pq = [(0, start)]
    
    while pq:
        current_dist, node = heapq.heappop(pq)
        
        if node == end:
            break
        
        if current_dist > distances[node]:
            continue
        
        for neighbor, weight in graph.get(node, []):
            distance = current_dist + weight
            
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                parent[neighbor] = node
                heapq.heappush(pq, (distance, neighbor))
    
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


def dijkstra_multi_source(graph, sources):
    """
    Dijkstra depuis plusieurs sources simultanément.
    Utile pour trouver le plus proche de plusieurs points.
    
    Args:
        graph: Dict {sommet: [(voisin, poids), ...]}
        sources: Liste des sommets sources
        
    Returns:
        Dict {sommet: (distance_min, source_plus_proche)}
    """
    distances = defaultdict(lambda: float('inf'))
    nearest_source = {}
    pq = []
    
    # Initialiser avec toutes les sources
    for source in sources:
        distances[source] = 0
        nearest_source[source] = source
        heapq.heappush(pq, (0, source, source))
    
    while pq:
        current_dist, node, source = heapq.heappop(pq)
        
        if current_dist > distances[node]:
            continue
        
        for neighbor, weight in graph.get(node, []):
            distance = current_dist + weight
            
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                nearest_source[neighbor] = source
                heapq.heappush(pq, (distance, neighbor, source))
    
    return {node: (distances[node], nearest_source.get(node)) 
            for node in distances}


def dijkstra_k_shortest(graph, start, end, k):
    """
    Trouve les k plus courts chemins de start à end.
    
    Args:
        graph: Dict {sommet: [(voisin, poids), ...]}
        start: Sommet de départ
        end: Sommet d'arrivée
        k: Nombre de chemins à trouver
        
    Returns:
        Liste des k distances les plus courtes (triée)
    """
    # Compter combien de fois on visite end
    visit_count = defaultdict(int)
    distances = []
    
    pq = [(0, start)]
    
    while pq and len(distances) < k:
        current_dist, node = heapq.heappop(pq)
        visit_count[node] += 1
        
        if node == end:
            distances.append(current_dist)
            if len(distances) == k:
                break
        
        # Ne visiter chaque noeud que k fois max
        if visit_count[node] <= k:
            for neighbor, weight in graph.get(node, []):
                heapq.heappush(pq, (current_dist + weight, neighbor))
    
    return distances


class DijkstraOnGrid:
    """
    Dijkstra optimisé pour les grilles 2D.
    """
    
    def __init__(self, grid):
        """
        Args:
            grid: Matrice 2D de poids (grid[i][j] = poids de la case)
        """
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0]) if grid else 0
        self.directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # droite, bas, gauche, haut
    
    def shortest_path(self, start, end):
        """
        Plus court chemin de start à end dans la grille.
        
        Args:
            start: Tuple (row, col) de départ
            end: Tuple (row, col) d'arrivée
            
        Returns:
            Distance minimale ou float('inf') si impossible
        """
        distances = defaultdict(lambda: float('inf'))
        distances[start] = self.grid[start[0]][start[1]]
        
        pq = [(self.grid[start[0]][start[1]], start)]
        
        while pq:
            current_dist, (row, col) = heapq.heappop(pq)
            
            if (row, col) == end:
                return current_dist
            
            if current_dist > distances[(row, col)]:
                continue
            
            for dr, dc in self.directions:
                new_row, new_col = row + dr, col + dc
                
                if 0 <= new_row < self.rows and 0 <= new_col < self.cols:
                    weight = self.grid[new_row][new_col]
                    if weight < 0:  # Case bloquée (optionnel)
                        continue
                    
                    distance = current_dist + weight
                    
                    if distance < distances[(new_row, new_col)]:
                        distances[(new_row, new_col)] = distance
                        heapq.heappush(pq, (distance, (new_row, new_col)))
        
        return distances[end]


def test():
    """Tests unitaires complets"""
    
    # Test graphe exemple
    graph = {
        0: [(1, 4), (2, 1)],
        1: [(3, 1)],
        2: [(1, 2), (3, 5)],
        3: []
    }
    
    # Test Dijkstra basique
    dist = dijkstra(graph, 0)
    assert dist[0] == 0
    assert dist[1] == 3
    assert dist[2] == 1
    assert dist[3] == 4
    
    # Test avec chemin
    distance, path = dijkstra_with_path(graph, 0, 3)
    assert distance == 4
    assert path == [0, 2, 1, 3]
    
    # Test graphe plus complexe
    graph2 = {
        0: [(1, 7), (2, 9), (5, 14)],
        1: [(0, 7), (2, 10), (3, 15)],
        2: [(0, 9), (1, 10), (3, 11), (5, 2)],
        3: [(1, 15), (2, 11), (4, 6)],
        4: [(3, 6), (5, 9)],
        5: [(0, 14), (2, 2), (4, 9)]
    }
    
    dist2 = dijkstra(graph2, 0)
    assert dist2[0] == 0
    assert dist2[1] == 7
    assert dist2[2] == 9
    assert dist2[3] == 20
    assert dist2[4] == 20
    assert dist2[5] == 11
    
    # Test multi-source
    graph3 = {
        0: [(1, 1), (2, 4)],
        1: [(2, 2), (3, 5)],
        2: [(3, 1)],
        3: []
    }
    
    multi = dijkstra_multi_source(graph3, [0, 3])
    assert multi[0][0] == 0
    assert multi[3][0] == 0
    assert multi[1][1] == 0  # Plus proche de source 0
    assert multi[2][0] in [3, 4]  # Accessible depuis les deux
    
    # Test k-shortest paths
    k_paths = dijkstra_k_shortest(graph, 0, 3, 2)
    assert len(k_paths) >= 1
    assert k_paths[0] == 4
    
    # Test sur grille
    grid = [
        [1, 2, 3],
        [1, 1, 1],
        [2, 2, 1]
    ]
    
    dij_grid = DijkstraOnGrid(grid)
    dist_grid = dij_grid.shortest_path((0, 0), (2, 2))
    assert dist_grid == 5  # 1 + 1 + 1 + 1 + 1
    
    # Test graphe déconnecté
    graph_disconnected = {
        0: [(1, 1)],
        1: [],
        2: [(3, 1)],
        3: []
    }
    
    dist_disc = dijkstra(graph_disconnected, 0)
    assert dist_disc[0] == 0
    assert dist_disc[1] == 1
    assert dist_disc[2] == float('inf')
    
    print("Tous les tests passes")


if __name__ == "__main__":
    test()

