"""
================================================================================
STOER-WAGNER MINIMUM CUT
================================================================================

Description:
-----------
Algorithme de Stoer-Wagner pour trouver la coupe minimale (min cut) dans
un graphe non-oriente pondere. Trouve la coupe qui minimise la somme des
poids des aretes coupees, separant le graphe en deux composantes.

Complexite:
-----------
- Temps: O(V^3) ou O(V * E + V^2 log V) avec heap
- Espace: O(V^2)

Cas d'usage typiques:
--------------------
1. Network reliability
2. Image segmentation
3. Clustering
4. Circuit design

Problemes classiques:
--------------------
- SPOJ MINCUT - Minimum Cut
- Codeforces 343E - Pumping Stations
- USACO Training - Drainage Ditches
- AtCoder ABC 239G - Builder Takahashi

Auteur: Assistant CP
Date: 2025
================================================================================
"""

from typing import List, Set, Tuple
import sys


class StoerWagner:
    """
    Stoer-Wagner pour Min Cut dans graphe non-oriente.
    
    Exemple:
    --------
    >>> sw = StoerWagner(4)
    >>> sw.add_edge(0, 1, 3)
    >>> sw.add_edge(1, 2, 2)
    >>> sw.add_edge(2, 3, 3)
    >>> sw.add_edge(3, 0, 2)
    >>> sw.add_edge(0, 2, 1)
    >>> 
    >>> min_cut = sw.min_cut()
    >>> print(f"Min cut: {min_cut}")
    """
    
    def __init__(self, n: int):
        """
        Args:
            n: Nombre de noeuds
        """
        self.n = n
        # Matrice d'adjacence (poids)
        self.adj = [[0] * n for _ in range(n)]
    
    def add_edge(self, u: int, v: int, weight: int):
        """
        Ajoute une arete non-orientee.
        
        Args:
            u, v: Noeuds
            weight: Poids de l'arete
        """
        self.adj[u][v] += weight
        self.adj[v][u] += weight
    
    def _minimum_cut_phase(self, nodes: Set[int]) -> Tuple[int, int, int]:
        """
        Une phase de l'algorithme de Stoer-Wagner.
        
        Returns:
            (cut_weight, last_node, second_last_node)
        """
        n = len(nodes)
        if n < 2:
            return float('inf'), -1, -1
        
        nodes_list = list(nodes)
        
        # Poids de connexion au set A
        weight_to_A = {}
        for node in nodes_list:
            weight_to_A[node] = 0
        
        # Demarre avec un noeud arbitraire
        A = set()
        start_node = nodes_list[0]
        
        last = -1
        second_last = -1
        
        for _ in range(n):
            # Trouve noeud avec poids maximum vers A
            max_weight = -1
            next_node = -1
            
            for node in nodes:
                if node not in A and weight_to_A[node] > max_weight:
                    max_weight = weight_to_A[node]
                    next_node = node
            
            if next_node == -1:
                break
            
            A.add(next_node)
            second_last = last
            last = next_node
            
            # Met a jour les poids
            for neighbor in nodes:
                if neighbor not in A:
                    weight_to_A[neighbor] += self.adj[next_node][neighbor]
        
        cut_weight = weight_to_A[last]
        
        return cut_weight, last, second_last
    
    def _merge_nodes(self, nodes: Set[int], u: int, v: int) -> Set[int]:
        """
        Fusionne deux noeuds u et v.
        
        Returns:
            Nouveau set de noeuds
        """
        # Ajoute les aretes de v a u
        for i in range(self.n):
            if i != u and i != v:
                self.adj[u][i] += self.adj[v][i]
                self.adj[i][u] += self.adj[i][v]
        
        # Retire v
        new_nodes = nodes.copy()
        new_nodes.remove(v)
        
        return new_nodes
    
    def min_cut(self) -> int:
        """
        Calcule le poids de la coupe minimale.
        
        Returns:
            Poids de la coupe minimale
        """
        nodes = set(range(self.n))
        min_cut_weight = float('inf')
        
        while len(nodes) > 1:
            cut_weight, last, second_last = self._minimum_cut_phase(nodes)
            
            if cut_weight < min_cut_weight:
                min_cut_weight = cut_weight
            
            # Fusionne last et second_last
            if second_last != -1:
                nodes = self._merge_nodes(nodes, second_last, last)
        
        return min_cut_weight


def minimum_cut_simple(n: int, edges: List[Tuple[int, int, int]]) -> int:
    """
    Interface simplifiee pour min cut.
    
    Args:
        n: Nombre de noeuds
        edges: Liste de (u, v, weight)
        
    Returns:
        Poids de la coupe minimale
        
    Exemple:
    --------
    >>> edges = [(0, 1, 3), (1, 2, 2), (2, 3, 3), (3, 0, 2), (0, 2, 1)]
    >>> result = minimum_cut_simple(4, edges)
    >>> print(result)  # 4
    """
    sw = StoerWagner(n)
    
    for u, v, w in edges:
        sw.add_edge(u, v, w)
    
    return sw.min_cut()


class KargerMinCut:
    """
    Algorithme de Karger (randomise) pour Min Cut.
    Plus simple mais probabiliste.
    
    Time: O(V^2 * log V) avec haute probabilite pour reussite
    """
    
    def __init__(self, n: int):
        self.n = n
        self.edges = []
    
    def add_edge(self, u: int, v: int, weight: int = 1):
        for _ in range(weight):
            self.edges.append((u, v))
    
    def min_cut(self, iterations: int = None) -> int:
        """
        Trouve min cut avec algorithme de Karger.
        
        Args:
            iterations: Nombre d'iterations (defaut: V^2 * log V)
            
        Returns:
            Poids de la coupe minimale (estime)
        """
        import random
        
        if iterations is None:
            iterations = self.n * self.n * 10
        
        min_cut_size = float('inf')
        
        for _ in range(iterations):
            cut_size = self._karger_iteration()
            min_cut_size = min(min_cut_size, cut_size)
        
        return min_cut_size
    
    def _karger_iteration(self) -> int:
        """Une iteration de Karger"""
        import random
        
        # Union-Find
        parent = list(range(self.n))
        rank = [0] * self.n
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px == py:
                return False
            if rank[px] < rank[py]:
                px, py = py, px
            parent[py] = px
            if rank[px] == rank[py]:
                rank[px] += 1
            return True
        
        edges = self.edges.copy()
        random.shuffle(edges)
        
        num_components = self.n
        
        # Contract jusqu'a 2 composantes
        for u, v in edges:
            if num_components <= 2:
                break
            
            if find(u) != find(v):
                union(u, v)
                num_components -= 1
        
        # Compte aretes entre les 2 composantes
        cut_size = 0
        for u, v in self.edges:
            if find(u) != find(v):
                cut_size += 1
        
        return cut_size


# ============================================================================
# TESTS UNITAIRES
# ============================================================================

def test_stoer_wagner_basic():
    """Test basic Stoer-Wagner"""
    sw = StoerWagner(4)
    
    # Graphe simple
    sw.add_edge(0, 1, 3)
    sw.add_edge(1, 2, 2)
    sw.add_edge(2, 3, 3)
    sw.add_edge(3, 0, 2)
    sw.add_edge(0, 2, 1)
    
    min_cut = sw.min_cut()
    
    assert min_cut == 4  # Coupe optimale
    
    print("✓ Test Stoer-Wagner basic passed")


def test_stoer_wagner_complete_graph():
    """Test sur graphe complet"""
    n = 4
    sw = StoerWagner(n)
    
    # K4 avec poids 1
    for i in range(n):
        for j in range(i + 1, n):
            sw.add_edge(i, j, 1)
    
    min_cut = sw.min_cut()
    
    # Min cut dans K4 = 3 (degre minimum)
    assert min_cut == 3
    
    print("✓ Test Stoer-Wagner complete graph passed")


def test_minimum_cut_simple():
    """Test interface simplifiee"""
    edges = [(0, 1, 3), (1, 2, 2), (2, 3, 3), (3, 0, 2), (0, 2, 1)]
    result = minimum_cut_simple(4, edges)
    
    assert result == 4
    
    print("✓ Test minimum cut simple passed")


def test_karger_min_cut():
    """Test Karger (probabiliste)"""
    kg = KargerMinCut(4)
    
    # Meme graphe
    kg.add_edge(0, 1, 3)
    kg.add_edge(1, 2, 2)
    kg.add_edge(2, 3, 3)
    kg.add_edge(3, 0, 2)
    kg.add_edge(0, 2, 1)
    
    min_cut = kg.min_cut(iterations=100)
    
    # Avec haute probabilite, trouve 4
    assert min_cut <= 5  # Tolerance pour randomness
    
    print("✓ Test Karger min cut passed")


# ============================================================================
# BENCHMARK
# ============================================================================

def benchmark_stoer_wagner():
    """Benchmark Stoer-Wagner"""
    import time
    import random
    
    print("\n=== Benchmark Stoer-Wagner Min Cut ===")
    
    for n in [10, 20, 30]:
        sw = StoerWagner(n)
        
        # Graphe dense aleatoire
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.5:
                    sw.add_edge(i, j, random.randint(1, 10))
        
        start = time.time()
        min_cut = sw.min_cut()
        elapsed = time.time() - start
        
        print(f"n={n:2d}: min_cut={min_cut:3d}, time={elapsed*1000:6.2f}ms")


if __name__ == "__main__":
    test_stoer_wagner_basic()
    test_stoer_wagner_complete_graph()
    test_minimum_cut_simple()
    test_karger_min_cut()
    
    benchmark_stoer_wagner()
    
    print("\n✓ Tous les tests Min Cut passes!")

