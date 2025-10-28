"""
================================================================================
HEAVY-LIGHT DECOMPOSITION (HLD)
================================================================================

Description:
-----------
Heavy-Light Decomposition decompose un arbre en chemins "lourds" et "legers"
permettant de repondre a des requetes sur chemins/sous-arbres en O(log^2 n).

Complexite:
-----------
- Construction: O(n)
- Query sur chemin: O(log^2 n) avec Segment Tree
- Update sur chemin: O(log^2 n) avec Segment Tree
- Espace: O(n)

Cas d'usage typiques:
--------------------
1. Somme/Min/Max sur chemin entre deux noeuds
2. Update de valeurs sur un chemin
3. LCA avec queries complexes
4. Problemes sur arbres avec path queries

Problemes classiques:
--------------------
- Codeforces 343D - Water Tree
- SPOJ QTREE - Query on a Tree
- HackerRank - Heavy Light White Falcon
- AtCoder ABC 138F - Coincidence

Auteur: Assistant CP
Date: 2025
================================================================================
"""

from typing import List, Callable, Any


class SegmentTree:
    """Segment Tree generique pour HLD"""
    
    def __init__(self, n: int, merge_fn: Callable, identity: Any):
        """
        Args:
            n: Taille du tableau
            merge_fn: Fonction de merge (ex: min, max, sum)
            identity: Element neutre (0 pour sum, inf pour min, -inf pour max)
        """
        self.n = n
        self.merge = merge_fn
        self.identity = identity
        self.tree = [identity] * (4 * n)
    
    def build(self, arr: List[Any], node: int = 1, start: int = 0, end: int = None):
        """Construit le segment tree depuis un tableau"""
        if end is None:
            end = self.n - 1
        
        if start == end:
            if start < len(arr):
                self.tree[node] = arr[start]
            return
        
        mid = (start + end) // 2
        self.build(arr, 2*node, start, mid)
        self.build(arr, 2*node+1, mid+1, end)
        self.tree[node] = self.merge(self.tree[2*node], self.tree[2*node+1])
    
    def update(self, idx: int, value: Any, node: int = 1, start: int = 0, end: int = None):
        """Update la valeur a l'indice idx"""
        if end is None:
            end = self.n - 1
        
        if start == end:
            self.tree[node] = value
            return
        
        mid = (start + end) // 2
        if idx <= mid:
            self.update(idx, value, 2*node, start, mid)
        else:
            self.update(idx, value, 2*node+1, mid+1, end)
        
        self.tree[node] = self.merge(self.tree[2*node], self.tree[2*node+1])
    
    def query(self, l: int, r: int, node: int = 1, start: int = 0, end: int = None) -> Any:
        """Query sur l'intervalle [l, r]"""
        if end is None:
            end = self.n - 1
        
        if r < start or l > end:
            return self.identity
        
        if l <= start and end <= r:
            return self.tree[node]
        
        mid = (start + end) // 2
        left_val = self.query(l, r, 2*node, start, mid)
        right_val = self.query(l, r, 2*node+1, mid+1, end)
        
        return self.merge(left_val, right_val)


class HeavyLightDecomposition:
    """
    Heavy-Light Decomposition pour queries sur arbres.
    
    Exemple:
    --------
    >>> # Arbre: 0-1-2-3
    >>> #           |
    >>> #           4
    >>> adj = [[] for _ in range(5)]
    >>> edges = [(0,1), (1,2), (2,3), (1,4)]
    >>> for u, v in edges:
    ...     adj[u].append(v)
    ...     adj[v].append(u)
    >>> 
    >>> values = [1, 2, 3, 4, 5]
    >>> hld = HeavyLightDecomposition(5, adj, values)
    >>> 
    >>> # Somme sur chemin 3->4
    >>> print(hld.query_path(3, 4))  # 3+2+2+5 = 14 (pas de double comptage du LCA)
    """
    
    def __init__(self, n: int, adj: List[List[int]], values: List[int], 
                 root: int = 0, merge_fn: Callable = None, identity: Any = None):
        """
        Args:
            n: Nombre de noeuds
            adj: Liste d'adjacence
            values: Valeurs des noeuds
            root: Racine de l'arbre
            merge_fn: Fonction de merge (defaut: sum)
            identity: Element neutre (defaut: 0)
        """
        self.n = n
        self.adj = adj
        self.root = root
        self.values = values
        
        # Parametres du segment tree
        self.merge_fn = merge_fn if merge_fn else lambda a, b: a + b
        self.identity = identity if identity is not None else 0
        
        # Arrays pour HLD
        self.parent = [-1] * n
        self.depth = [0] * n
        self.heavy = [-1] * n  # Fils lourd (-1 si feuille)
        self.head = list(range(n))  # Tete de chaine
        self.pos = [0] * n  # Position dans l'ordre HLD
        
        self.subtree_size = [0] * n
        self.cur_pos = 0
        
        # Construction
        self._dfs_size(root, -1)
        self._dfs_hld(root, -1)
        
        # Segment tree sur l'ordre HLD
        reordered_values = [0] * n
        for i in range(n):
            reordered_values[self.pos[i]] = values[i]
        
        self.seg_tree = SegmentTree(n, self.merge_fn, self.identity)
        self.seg_tree.build(reordered_values)
    
    def _dfs_size(self, u: int, p: int):
        """Calcule taille des sous-arbres et trouve fils lourd"""
        self.subtree_size[u] = 1
        self.parent[u] = p
        
        max_subtree = 0
        for v in self.adj[u]:
            if v == p:
                continue
            
            self.depth[v] = self.depth[u] + 1
            self._dfs_size(v, u)
            self.subtree_size[u] += self.subtree_size[v]
            
            # Le fils avec le plus grand sous-arbre devient le fils lourd
            if self.subtree_size[v] > max_subtree:
                max_subtree = self.subtree_size[v]
                self.heavy[u] = v
    
    def _dfs_hld(self, u: int, p: int):
        """Decompose l'arbre en chaines lourdes"""
        self.pos[u] = self.cur_pos
        self.cur_pos += 1
        
        # Traiter d'abord le fils lourd (meme chaine)
        if self.heavy[u] != -1:
            self.head[self.heavy[u]] = self.head[u]
            self._dfs_hld(self.heavy[u], u)
        
        # Puis les fils legers (nouvelles chaines)
        for v in self.adj[u]:
            if v == p or v == self.heavy[u]:
                continue
            self._dfs_hld(v, u)
    
    def lca(self, u: int, v: int) -> int:
        """Trouve le LCA de u et v en O(log n)"""
        while self.head[u] != self.head[v]:
            if self.depth[self.head[u]] < self.depth[self.head[v]]:
                u, v = v, u
            u = self.parent[self.head[u]]
        
        return u if self.depth[u] < self.depth[v] else v
    
    def query_path(self, u: int, v: int) -> Any:
        """
        Query sur le chemin de u a v.
        
        Time: O(log^2 n)
        
        Args:
            u, v: Noeuds extremites
            
        Returns:
            Resultat de la fonction merge sur le chemin
        """
        result = self.identity
        
        while self.head[u] != self.head[v]:
            if self.depth[self.head[u]] < self.depth[self.head[v]]:
                u, v = v, u
            
            # Query sur la chaine de u jusqu'a sa tete
            result = self.merge_fn(
                result,
                self.seg_tree.query(self.pos[self.head[u]], self.pos[u])
            )
            u = self.parent[self.head[u]]
        
        # Query sur la chaine finale
        if self.depth[u] > self.depth[v]:
            u, v = v, u
        result = self.merge_fn(
            result,
            self.seg_tree.query(self.pos[u], self.pos[v])
        )
        
        return result
    
    def update_node(self, u: int, value: Any):
        """
        Update la valeur d'un noeud.
        
        Time: O(log n)
        
        Args:
            u: Noeud a updater
            value: Nouvelle valeur
        """
        self.values[u] = value
        self.seg_tree.update(self.pos[u], value)
    
    def query_subtree(self, u: int) -> Any:
        """
        Query sur le sous-arbre enracine en u.
        
        Time: O(log n)
        
        Args:
            u: Racine du sous-arbre
            
        Returns:
            Resultat de la fonction merge sur le sous-arbre
        """
        return self.seg_tree.query(
            self.pos[u],
            self.pos[u] + self.subtree_size[u] - 1
        )


def solve_path_sum_queries(n: int, edges: List[tuple], values: List[int],
                           queries: List[tuple]) -> List[int]:
    """
    Resout des queries de somme sur chemins.
    
    Args:
        n: Nombre de noeuds
        edges: Aretes de l'arbre (u, v)
        values: Valeurs des noeuds
        queries: Liste de (type, u, v) ou (type, u, new_val)
                 type=0: query somme sur chemin u->v
                 type=1: update noeud u a new_val
    
    Returns:
        Liste des reponses aux queries de type 0
    """
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    hld = HeavyLightDecomposition(n, adj, values)
    
    results = []
    for query in queries:
        if query[0] == 0:  # Query
            u, v = query[1], query[2]
            results.append(hld.query_path(u, v))
        else:  # Update
            u, new_val = query[1], query[2]
            hld.update_node(u, new_val)
    
    return results


# ============================================================================
# TESTS UNITAIRES
# ============================================================================

def test_hld_basic():
    """Test basic HLD sur petit arbre"""
    # Arbre: 0-1-2
    adj = [[] for _ in range(3)]
    edges = [(0, 1), (1, 2)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    values = [1, 2, 3]
    hld = HeavyLightDecomposition(3, adj, values)
    
    # Somme 0->2
    assert hld.query_path(0, 2) == 6  # 1+2+3
    # Somme 0->1
    assert hld.query_path(0, 1) == 3  # 1+2
    # Somme 1->2
    assert hld.query_path(1, 2) == 5  # 2+3
    
    print("✓ Test HLD basic passed")


def test_hld_update():
    """Test updates avec HLD"""
    adj = [[] for _ in range(3)]
    edges = [(0, 1), (1, 2)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    values = [1, 2, 3]
    hld = HeavyLightDecomposition(3, adj, values)
    
    # Update noeud 1
    hld.update_node(1, 10)
    assert hld.query_path(0, 2) == 14  # 1+10+3
    
    print("✓ Test HLD update passed")


def test_hld_lca():
    """Test LCA avec HLD"""
    # Arbre:    0
    #          / \
    #         1   2
    #        / \
    #       3   4
    adj = [[] for _ in range(5)]
    edges = [(0, 1), (0, 2), (1, 3), (1, 4)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    values = [1] * 5
    hld = HeavyLightDecomposition(5, adj, values)
    
    assert hld.lca(3, 4) == 1
    assert hld.lca(3, 2) == 0
    assert hld.lca(0, 4) == 0
    
    print("✓ Test HLD LCA passed")


def test_hld_subtree():
    """Test queries sur sous-arbres"""
    # Arbre:    0
    #          / \
    #         1   2
    #        /
    #       3
    adj = [[] for _ in range(4)]
    edges = [(0, 1), (0, 2), (1, 3)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    values = [1, 2, 3, 4]
    hld = HeavyLightDecomposition(4, adj, values)
    
    # Sous-arbre de 1: noeuds 1,3 = 2+4 = 6
    assert hld.query_subtree(1) == 6
    # Sous-arbre de 0: tous = 10
    assert hld.query_subtree(0) == 10
    
    print("✓ Test HLD subtree passed")


def test_hld_with_max():
    """Test HLD avec fonction max"""
    adj = [[] for _ in range(4)]
    edges = [(0, 1), (1, 2), (2, 3)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    values = [1, 5, 2, 8]
    hld = HeavyLightDecomposition(
        4, adj, values,
        merge_fn=max,
        identity=float('-inf')
    )
    
    # Max sur chemin 0->3
    assert hld.query_path(0, 3) == 8
    # Max sur chemin 0->1
    assert hld.query_path(0, 1) == 5
    
    print("✓ Test HLD with max passed")


# ============================================================================
# BENCHMARK
# ============================================================================

def benchmark_hld():
    """Benchmark HLD sur grand arbre"""
    import time
    import random
    
    n = 10000
    
    # Genere un arbre aleatoire (chaine)
    adj = [[] for _ in range(n)]
    for i in range(n-1):
        adj[i].append(i+1)
        adj[i+1].append(i)
    
    values = [random.randint(1, 100) for _ in range(n)]
    
    # Construction
    start = time.time()
    hld = HeavyLightDecomposition(n, adj, values)
    build_time = time.time() - start
    
    # Queries
    num_queries = 1000
    queries = [(random.randint(0, n-1), random.randint(0, n-1)) 
               for _ in range(num_queries)]
    
    start = time.time()
    for u, v in queries:
        hld.query_path(u, v)
    query_time = time.time() - start
    
    print(f"\n=== Benchmark HLD (n={n}) ===")
    print(f"Build time: {build_time*1000:.2f}ms")
    print(f"Query time: {query_time*1000:.2f}ms for {num_queries} queries")
    print(f"Avg query: {query_time/num_queries*1000:.3f}ms")


if __name__ == "__main__":
    # Tests
    test_hld_basic()
    test_hld_update()
    test_hld_lca()
    test_hld_subtree()
    test_hld_with_max()
    
    # Benchmark
    benchmark_hld()
    
    print("\n✓ Tous les tests HLD passes!")

