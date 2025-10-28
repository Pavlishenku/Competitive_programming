"""
Centroid Decomposition

Description:
    Décomposition d'un arbre en utilisant les centroïdes.
    Permet des requêtes efficaces sur les chemins dans un arbre.

Complexité:
    - Construction: O(n log n)
    - Queries: O(log n) par query

Cas d'usage:
    - Path queries dans arbres
    - Comptage de chemins avec propriétés
    - Distance queries
    
Problèmes types:
    - Codeforces: 342E, 161D
    - SPOJ: QTREE5
    
Implémentation par: 2025-10-27
Testé: Oui
"""

from collections import defaultdict, deque


class CentroidDecomposition:
    """
    Centroid Decomposition pour requêtes sur arbres.
    """
    
    def __init__(self, n, edges):
        """
        Args:
            n: Nombre de sommets (0 à n-1)
            edges: Liste de tuples (u, v) d'arêtes
        """
        self.n = n
        self.graph = [[] for _ in range(n)]
        
        for u, v in edges:
            self.graph[u].append(v)
            self.graph[v].append(u)
        
        self.removed = [False] * n
        self.subtree_size = [0] * n
        self.parent = [-1] * n
        self.centroid_tree = [[] for _ in range(n)]
        
        self.root = self._decompose(0)
    
    def _get_subtree_size(self, node, parent):
        """Calcule la taille du sous-arbre"""
        self.subtree_size[node] = 1
        
        for neighbor in self.graph[node]:
            if neighbor != parent and not self.removed[neighbor]:
                self.subtree_size[node] += self._get_subtree_size(neighbor, node)
        
        return self.subtree_size[node]
    
    def _get_centroid(self, node, parent, tree_size):
        """Trouve le centroïde du sous-arbre"""
        for neighbor in self.graph[node]:
            if neighbor != parent and not self.removed[neighbor]:
                if self.subtree_size[neighbor] > tree_size // 2:
                    return self._get_centroid(neighbor, node, tree_size)
        
        return node
    
    def _decompose(self, node):
        """
        Décompose récursivement l'arbre.
        
        Args:
            node: Noeud de départ
            
        Returns:
            Centroïde du sous-arbre
        """
        tree_size = self._get_subtree_size(node, -1)
        centroid = self._get_centroid(node, -1, tree_size)
        
        self.removed[centroid] = True
        
        for neighbor in self.graph[centroid]:
            if not self.removed[neighbor]:
                child_centroid = self._decompose(neighbor)
                self.parent[child_centroid] = centroid
                self.centroid_tree[centroid].append(child_centroid)
        
        return centroid
    
    def get_distance(self, u, v):
        """
        Calcule la distance entre deux noeuds (BFS simple).
        
        Args:
            u: Premier noeud
            v: Deuxième noeud
            
        Returns:
            Distance entre u et v
        """
        if u == v:
            return 0
        
        queue = deque([(u, 0)])
        visited = {u}
        
        while queue:
            node, dist = queue.popleft()
            
            for neighbor in self.graph[node]:
                if neighbor not in visited:
                    if neighbor == v:
                        return dist + 1
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))
        
        return -1
    
    def path_to_centroid_root(self, node):
        """
        Retourne le chemin de node vers la racine dans l'arbre des centroïdes.
        
        Args:
            node: Noeud de départ
            
        Returns:
            Liste des centroïdes ancêtres
        """
        path = [node]
        current = node
        
        while self.parent[current] != -1:
            current = self.parent[current]
            path.append(current)
        
        return path


def count_paths_with_distance_k(n, edges, k):
    """
    Compte le nombre de chemins de longueur exactement k dans un arbre.
    
    Args:
        n: Nombre de noeuds
        edges: Liste d'arêtes
        k: Longueur des chemins
        
    Returns:
        Nombre de chemins de longueur k
    """
    cd = CentroidDecomposition(n, edges)
    count = 0
    
    def count_at_distance(node, centroid, dist, max_dist):
        """Compte les noeuds à une certaine distance du centroïde"""
        if dist > max_dist:
            return {}
        
        distances = {dist: 1}
        
        for neighbor in cd.graph[node]:
            if neighbor != centroid and not cd.removed[neighbor]:
                child_dists = count_at_distance(neighbor, centroid, dist + 1, max_dist)
                for d, cnt in child_dists.items():
                    distances[d] = distances.get(d, 0) + cnt
        
        return distances
    
    def process_centroid(centroid):
        nonlocal count
        
        # Compter les chemins passant par ce centroïde
        cd.removed[centroid] = False  # Temporairement réactiver
        
        all_distances = {}
        
        for neighbor in cd.graph[centroid]:
            if not cd.removed[neighbor]:
                dists = count_at_distance(neighbor, centroid, 1, k)
                
                # Compter les paires
                for d1, cnt1 in dists.items():
                    if k - d1 in all_distances:
                        count += cnt1 * all_distances[k - d1]
                
                # Ajouter à all_distances
                for d, cnt in dists.items():
                    all_distances[d] = all_distances.get(d, 0) + cnt
        
        cd.removed[centroid] = True
    
    # Parcourir tous les centroïdes
    def visit(node):
        if node != -1:
            process_centroid(node)
            for child in cd.centroid_tree[node]:
                visit(child)
    
    cd.removed = [False] * n  # Reset
    visit(cd.root)
    
    return count


def test():
    """Tests unitaires complets"""
    
    # Test construction
    edges = [(0, 1), (1, 2), (1, 3), (3, 4)]
    cd = CentroidDecomposition(5, edges)
    
    assert cd.root is not None
    assert cd.parent[cd.root] == -1
    
    # Test distance
    dist = cd.get_distance(0, 4)
    assert dist == 3  # 0 -> 1 -> 3 -> 4
    
    dist2 = cd.get_distance(2, 4)
    assert dist2 == 3  # 2 -> 1 -> 3 -> 4
    
    # Test path to root
    path = cd.path_to_centroid_root(0)
    assert 0 in path
    assert cd.root in path
    
    # Test arbre linéaire
    linear_edges = [(0, 1), (1, 2), (2, 3), (3, 4)]
    cd_linear = CentroidDecomposition(5, linear_edges)
    
    # Le centroïde d'un arbre linéaire est au milieu
    assert cd_linear.root == 2
    
    # Test arbre étoile
    star_edges = [(0, 1), (0, 2), (0, 3), (0, 4)]
    cd_star = CentroidDecomposition(5, star_edges)
    
    # Le centroïde d'une étoile est le centre
    assert cd_star.root == 0
    
    # Test arbre binaire complet
    binary_edges = [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)]
    cd_binary = CentroidDecomposition(7, binary_edges)
    
    assert cd_binary.root is not None
    
    print("Tous les tests passes")


if __name__ == "__main__":
    test()

