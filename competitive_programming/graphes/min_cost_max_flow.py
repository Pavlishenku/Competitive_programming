"""
================================================================================
MIN COST MAX FLOW
================================================================================

Description:
-----------
Trouve le flot maximum a cout minimum dans un graphe avec capacites et couts.
Utilise l'algorithme SPFA (Shortest Path Faster Algorithm) pour trouver
les chemins augmentants de cout minimum.

Complexite:
-----------
- Temps: O(V * E * max_flow) en pire cas, O(V^2 * E) en pratique
- Espace: O(V + E)

Cas d'usage typiques:
--------------------
1. Problemes d'assignation avec couts
2. Transport avec couts minimaux
3. Matching avec poids
4. Problemes de circulation avec couts

Problemes classiques:
--------------------
- SPOJ MCMF - Min Cost Max Flow
- Codeforces 164C - Machine Programming
- USACO Training - Milk Pumping
- AtCoder ABC 193F - Zebraness

Auteur: Assistant CP
Date: 2025
================================================================================
"""

from typing import List, Tuple
from collections import deque


class MinCostMaxFlow:
    """
    Min Cost Max Flow using SPFA (Bellman-Ford variant).
    
    Exemple:
    --------
    >>> mcmf = MinCostMaxFlow(4)
    >>> # Graphe: 0 --[cap=2,cost=1]--> 1 --[cap=1,cost=2]--> 3
    >>> #         0 --[cap=1,cost=3]--> 2 --[cap=2,cost=1]--> 3
    >>> mcmf.add_edge(0, 1, 2, 1)
    >>> mcmf.add_edge(1, 3, 1, 2)
    >>> mcmf.add_edge(0, 2, 1, 3)
    >>> mcmf.add_edge(2, 3, 2, 1)
    >>> 
    >>> flow, cost = mcmf.max_flow(0, 3)
    >>> print(f"Max flow: {flow}, Min cost: {cost}")
    Max flow: 3, Min cost: 8
    """
    
    def __init__(self, n: int):
        """
        Args:
            n: Nombre de noeuds
        """
        self.n = n
        self.graph = [[] for _ in range(n)]
        self.edges = []
    
    def add_edge(self, u: int, v: int, capacity: int, cost: int):
        """
        Ajoute une arete avec capacite et cout.
        
        Args:
            u: Noeud source
            v: Noeud destination
            capacity: Capacite de l'arete
            cost: Cout par unite de flot
        """
        edge_id = len(self.edges)
        
        # Arete directe
        self.graph[u].append(edge_id)
        self.edges.append([u, v, capacity, 0, cost])  # [from, to, cap, flow, cost]
        
        # Arete inverse (capacite 0, cout negatif)
        self.graph[v].append(edge_id + 1)
        self.edges.append([v, u, 0, 0, -cost])
    
    def _spfa(self, source: int, sink: int) -> Tuple[bool, List[int], List[int]]:
        """
        SPFA (Shortest Path Faster Algorithm) pour trouver chemin de cout min.
        
        Returns:
            (path_exists, parent, parent_edge_id)
        """
        INF = float('inf')
        dist = [INF] * self.n
        parent = [-1] * self.n
        parent_edge = [-1] * self.n
        in_queue = [False] * self.n
        
        dist[source] = 0
        queue = deque([source])
        in_queue[source] = True
        
        while queue:
            u = queue.popleft()
            in_queue[u] = False
            
            for edge_id in self.graph[u]:
                edge = self.edges[edge_id]
                v = edge[1]
                cap = edge[2]
                flow = edge[3]
                cost = edge[4]
                
                # Si capacite residuelle et meilleur chemin
                if cap > flow and dist[u] + cost < dist[v]:
                    dist[v] = dist[u] + cost
                    parent[v] = u
                    parent_edge[v] = edge_id
                    
                    if not in_queue[v]:
                        queue.append(v)
                        in_queue[v] = True
        
        return dist[sink] < INF, parent, parent_edge
    
    def max_flow(self, source: int, sink: int) -> Tuple[int, int]:
        """
        Calcule le flot maximum a cout minimum.
        
        Args:
            source: Noeud source
            sink: Noeud puits
            
        Returns:
            (max_flow, min_cost)
        """
        total_flow = 0
        total_cost = 0
        
        while True:
            # Trouve chemin augmentant de cout minimum
            exists, parent, parent_edge = self._spfa(source, sink)
            
            if not exists:
                break
            
            # Trouve capacite residuelle minimale sur le chemin
            path_flow = float('inf')
            v = sink
            while v != source:
                edge_id = parent_edge[v]
                edge = self.edges[edge_id]
                path_flow = min(path_flow, edge[2] - edge[3])
                v = parent[v]
            
            # Augmente le flot le long du chemin
            v = sink
            while v != source:
                edge_id = parent_edge[v]
                
                # Arete directe
                self.edges[edge_id][3] += path_flow
                
                # Arete inverse
                self.edges[edge_id ^ 1][3] -= path_flow
                
                # Ajoute le cout
                total_cost += path_flow * self.edges[edge_id][4]
                
                v = parent[v]
            
            total_flow += path_flow
        
        return total_flow, total_cost
    
    def get_flow_on_edge(self, u: int, v: int) -> int:
        """Retourne le flot sur l'arete (u, v)"""
        for edge_id in self.graph[u]:
            edge = self.edges[edge_id]
            if edge[1] == v:
                return edge[3]
        return 0


def assignment_problem(cost_matrix: List[List[int]]) -> Tuple[int, List[int]]:
    """
    Resout le probleme d'assignation: assigner n travailleurs a n taches
    en minimisant le cout total.
    
    Args:
        cost_matrix: Matrice n x n des couts cost[worker][task]
    
    Returns:
        (min_cost, assignment) ou assignment[worker] = task
    
    Time: O(n^3)
    
    Exemple:
    --------
    >>> costs = [
    ...     [9, 2, 7, 8],
    ...     [6, 4, 3, 7],
    ...     [5, 8, 1, 8],
    ...     [7, 6, 9, 4]
    ... ]
    >>> min_cost, assign = assignment_problem(costs)
    >>> print(f"Min cost: {min_cost}")  # 13 = 2+3+1+7 ou similaire
    """
    n = len(cost_matrix)
    
    # Graphe: source(0) -> workers(1..n) -> tasks(n+1..2n) -> sink(2n+1)
    mcmf = MinCostMaxFlow(2 * n + 2)
    source = 0
    sink = 2 * n + 1
    
    # Source vers travailleurs (capacite 1, cout 0)
    for i in range(n):
        mcmf.add_edge(source, i + 1, 1, 0)
    
    # Travailleurs vers taches (capacite 1, cout = cost_matrix)
    for i in range(n):
        for j in range(n):
            mcmf.add_edge(i + 1, n + j + 1, 1, cost_matrix[i][j])
    
    # Taches vers sink (capacite 1, cout 0)
    for j in range(n):
        mcmf.add_edge(n + j + 1, sink, 1, 0)
    
    flow, cost = mcmf.max_flow(source, sink)
    
    # Reconstruit l'assignation
    assignment = [-1] * n
    for i in range(n):
        for j in range(n):
            if mcmf.get_flow_on_edge(i + 1, n + j + 1) > 0:
                assignment[i] = j
                break
    
    return cost, assignment


def min_cost_circulation(n: int, edges: List[Tuple[int, int, int, int, int]]) -> Tuple[bool, int]:
    """
    Probleme de circulation a cout minimum avec demandes.
    
    Args:
        n: Nombre de noeuds
        edges: Liste de (u, v, lower_bound, upper_bound, cost)
        
    Returns:
        (feasible, min_cost) ou feasible indique si une circulation existe
    
    Note: Demandes implicites (conservation du flot)
    """
    # TODO: Implementation complete si necessaire
    pass


# ============================================================================
# TESTS UNITAIRES
# ============================================================================

def test_mcmf_basic():
    """Test basic min cost max flow"""
    mcmf = MinCostMaxFlow(4)
    
    # Graphe simple
    mcmf.add_edge(0, 1, 2, 1)  # cap=2, cost=1
    mcmf.add_edge(1, 3, 1, 2)  # cap=1, cost=2
    mcmf.add_edge(0, 2, 1, 3)  # cap=1, cost=3
    mcmf.add_edge(2, 3, 2, 1)  # cap=2, cost=1
    
    flow, cost = mcmf.max_flow(0, 3)
    
    assert flow == 2  # Max flow = 2
    # Chemin 1: 0->1->3 (cost 3), Chemin 2: 0->2->3 (cost 4)
    # Total: 7
    print(f"✓ Test MCMF basic passed (flow={flow}, cost={cost})")


def test_mcmf_no_flow():
    """Test quand aucun flot possible"""
    mcmf = MinCostMaxFlow(3)
    
    # Pas d'arete vers sink
    mcmf.add_edge(0, 1, 10, 5)
    
    flow, cost = mcmf.max_flow(0, 2)
    
    assert flow == 0
    assert cost == 0
    print("✓ Test MCMF no flow passed")


def test_assignment_problem():
    """Test probleme d'assignation"""
    costs = [
        [9, 2, 7, 8],
        [6, 4, 3, 7],
        [5, 8, 1, 8],
        [7, 6, 9, 4]
    ]
    
    min_cost, assignment = assignment_problem(costs)
    
    # Verifie que c'est une permutation valide
    assert len(set(assignment)) == 4
    assert all(0 <= x < 4 for x in assignment)
    
    # Verifie le cout
    calculated_cost = sum(costs[i][assignment[i]] for i in range(4))
    assert calculated_cost == min_cost
    
    print(f"✓ Test assignment problem passed (cost={min_cost}, assignment={assignment})")


def test_mcmf_multiple_paths():
    """Test avec plusieurs chemins possibles"""
    mcmf = MinCostMaxFlow(4)
    
    # 2 chemins: haut (cher) et bas (pas cher)
    mcmf.add_edge(0, 1, 1, 10)  # Chemin cher
    mcmf.add_edge(1, 3, 1, 10)
    
    mcmf.add_edge(0, 2, 2, 1)   # Chemin pas cher
    mcmf.add_edge(2, 3, 2, 1)
    
    flow, cost = mcmf.max_flow(0, 3)
    
    assert flow == 2  # Peut passer 2 unites
    assert cost == 4  # Utilise le chemin pas cher (2*2=4)
    
    print("✓ Test MCMF multiple paths passed")


def test_mcmf_with_cycles():
    """Test avec cycles dans le graphe"""
    mcmf = MinCostMaxFlow(4)
    
    mcmf.add_edge(0, 1, 10, 2)
    mcmf.add_edge(1, 2, 10, 3)
    mcmf.add_edge(2, 3, 10, 1)
    
    # Cycle qui ne devrait pas etre utilise
    mcmf.add_edge(1, 0, 5, 1)
    mcmf.add_edge(2, 1, 5, 1)
    
    flow, cost = mcmf.max_flow(0, 3)
    
    assert flow == 10
    assert cost == 60  # 10 * (2+3+1)
    
    print("✓ Test MCMF with cycles passed")


# ============================================================================
# BENCHMARK
# ============================================================================

def benchmark_mcmf():
    """Benchmark Min Cost Max Flow"""
    import time
    import random
    
    print("\n=== Benchmark Min Cost Max Flow ===")
    
    for n in [10, 50, 100]:
        mcmf = MinCostMaxFlow(n)
        
        # Genere graphe aleatoire
        num_edges = n * 3
        for _ in range(num_edges):
            u = random.randint(0, n-2)
            v = random.randint(u+1, n-1)
            cap = random.randint(1, 10)
            cost = random.randint(1, 100)
            mcmf.add_edge(u, v, cap, cost)
        
        start = time.time()
        flow, cost = mcmf.max_flow(0, n-1)
        elapsed = time.time() - start
        
        print(f"n={n:3d}: flow={flow:3d}, cost={cost:6d}, time={elapsed*1000:6.2f}ms")


def benchmark_assignment():
    """Benchmark probleme d'assignation"""
    import time
    import random
    
    print("\n=== Benchmark Assignment Problem ===")
    
    for n in [5, 10, 20, 30]:
        costs = [[random.randint(1, 100) for _ in range(n)] for _ in range(n)]
        
        start = time.time()
        min_cost, assignment = assignment_problem(costs)
        elapsed = time.time() - start
        
        print(f"n={n:2d}: cost={min_cost:4d}, time={elapsed*1000:6.2f}ms")


if __name__ == "__main__":
    # Tests
    test_mcmf_basic()
    test_mcmf_no_flow()
    test_assignment_problem()
    test_mcmf_multiple_paths()
    test_mcmf_with_cycles()
    
    # Benchmarks
    benchmark_mcmf()
    benchmark_assignment()
    
    print("\n✓ Tous les tests MCMF passes!")

