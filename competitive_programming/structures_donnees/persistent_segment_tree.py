"""
================================================================================
PERSISTENT SEGMENT TREE
================================================================================

Description:
-----------
Segment Tree persistant qui conserve toutes les versions anterieures.
Chaque update cree une nouvelle version sans modifier les anciennes.
Utilise path copying pour economiser l'espace: O(log n) par version.

Complexite:
-----------
- Update: O(log n) temps et espace
- Query: O(log n)
- Espace total: O(n + q log n) pour q updates

Cas d'usage typiques:
--------------------
1. Queries sur versions historiques
2. Problemes offline avec rollback
3. Functional data structures
4. Kth number in range (avec tri)

Problemes classiques:
--------------------
- SPOJ MKTHNUM - K-th Number
- Codeforces 707D - Persistent Bookcase
- CSES Range Queries - List Removals
- AtCoder ABC 218F - Blocked Roads

Auteur: Assistant CP
Date: 2025
================================================================================
"""

from typing import List, Optional, Callable


class Node:
    """Noeud du Persistent Segment Tree"""
    
    def __init__(self, value: int = 0):
        self.value = value
        self.left: Optional[Node] = None
        self.right: Optional[Node] = None


class PersistentSegmentTree:
    """
    Segment Tree Persistant.
    
    Exemple:
    --------
    >>> pst = PersistentSegmentTree(5)
    >>> 
    >>> # Version 0: [0, 0, 0, 0, 0]
    >>> v0 = pst.build([0] * 5)
    >>> 
    >>> # Version 1: update position 2 to 5
    >>> v1 = pst.update(v0, 0, 4, 2, 5)
    >>> 
    >>> # Version 2: update position 3 to 7
    >>> v2 = pst.update(v1, 0, 4, 3, 7)
    >>> 
    >>> # Query somme sur v1
    >>> print(pst.query(v1, 0, 4, 0, 4))  # 5
    >>> 
    >>> # Query somme sur v2
    >>> print(pst.query(v2, 0, 4, 0, 4))  # 12
    """
    
    def __init__(self, n: int, merge_fn: Callable = None, identity: int = 0):
        """
        Args:
            n: Taille du tableau
            merge_fn: Fonction de merge (defaut: sum)
            identity: Element neutre
        """
        self.n = n
        self.merge = merge_fn if merge_fn else lambda a, b: a + b
        self.identity = identity
    
    def build(self, arr: List[int]) -> Node:
        """
        Construit la version initiale.
        
        Args:
            arr: Tableau initial
            
        Returns:
            Racine de la version 0
        """
        return self._build(arr, 0, self.n - 1)
    
    def _build(self, arr: List[int], l: int, r: int) -> Node:
        """Construction recursive"""
        node = Node()
        
        if l == r:
            node.value = arr[l] if l < len(arr) else self.identity
            return node
        
        mid = (l + r) // 2
        node.left = self._build(arr, l, mid)
        node.right = self._build(arr, mid + 1, r)
        node.value = self.merge(node.left.value, node.right.value)
        
        return node
    
    def update(self, root: Node, l: int, r: int, pos: int, value: int) -> Node:
        """
        Cree une nouvelle version avec update en position pos.
        
        Time: O(log n)
        Space: O(log n)
        
        Args:
            root: Racine de la version courante
            l, r: Bornes actuelles
            pos: Position a updater
            value: Nouvelle valeur
            
        Returns:
            Racine de la nouvelle version
        """
        new_node = Node()
        
        if l == r:
            new_node.value = value
            return new_node
        
        mid = (l + r) // 2
        
        if pos <= mid:
            new_node.left = self.update(root.left, l, mid, pos, value)
            new_node.right = root.right
        else:
            new_node.left = root.left
            new_node.right = self.update(root.right, mid + 1, r, pos, value)
        
        new_node.value = self.merge(new_node.left.value, new_node.right.value)
        
        return new_node
    
    def query(self, root: Node, l: int, r: int, ql: int, qr: int) -> int:
        """
        Query sur une version.
        
        Time: O(log n)
        
        Args:
            root: Racine de la version
            l, r: Bornes actuelles
            ql, qr: Bornes de la query
            
        Returns:
            Resultat de la query
        """
        if qr < l or ql > r:
            return self.identity
        
        if ql <= l and r <= qr:
            return root.value
        
        mid = (l + r) // 2
        left_val = self.query(root.left, l, mid, ql, qr) if root.left else self.identity
        right_val = self.query(root.right, mid + 1, r, ql, qr) if root.right else self.identity
        
        return self.merge(left_val, right_val)


class PersistentSegmentTreeKthNumber:
    """
    Persistent Segment Tree pour trouver le k-ieme nombre dans un range.
    Utilise des arbres tries par valeur.
    
    Exemple:
    --------
    >>> arr = [3, 1, 4, 1, 5]
    >>> pst_kth = PersistentSegmentTreeKthNumber(arr)
    >>> 
    >>> # 2e plus petit nombre dans [0, 4]
    >>> print(pst_kth.kth_number(0, 4, 2))  # 3
    """
    
    def __init__(self, arr: List[int]):
        """
        Args:
            arr: Tableau d'entree
        """
        # Compression des coordonnees
        self.arr = arr
        self.n = len(arr)
        self.sorted_values = sorted(set(arr))
        self.value_to_idx = {v: i for i, v in enumerate(self.sorted_values)}
        
        self.m = len(self.sorted_values)
        
        # Construit versions
        self.versions = [None] * (self.n + 1)
        self.versions[0] = Node()  # Version vide
        
        for i in range(self.n):
            idx = self.value_to_idx[arr[i]]
            self.versions[i + 1] = self._insert(self.versions[i], 0, self.m - 1, idx)
    
    def _insert(self, root: Node, l: int, r: int, pos: int) -> Node:
        """Insere un element (incremente compteur)"""
        new_node = Node(root.value + 1 if root else 1)
        
        if l == r:
            return new_node
        
        mid = (l + r) // 2
        
        if pos <= mid:
            new_node.left = self._insert(
                root.left if root and root.left else Node(), 
                l, mid, pos
            )
            new_node.right = root.right if root else None
        else:
            new_node.left = root.left if root else None
            new_node.right = self._insert(
                root.right if root and root.right else Node(), 
                mid + 1, r, pos
            )
        
        left_val = new_node.left.value if new_node.left else 0
        right_val = new_node.right.value if new_node.right else 0
        new_node.value = left_val + right_val
        
        return new_node
    
    def kth_number(self, l: int, r: int, k: int) -> int:
        """
        Trouve le k-ieme plus petit nombre dans arr[l:r+1].
        
        Time: O(log n)
        
        Args:
            l, r: Indices du range (inclusive)
            k: Position (1-indexed, 1 = minimum)
            
        Returns:
            Le k-ieme plus petit nombre
        """
        return self.sorted_values[
            self._kth(self.versions[l], self.versions[r + 1], 0, self.m - 1, k)
        ]
    
    def _kth(self, left_root: Node, right_root: Node, l: int, r: int, k: int) -> int:
        """Trouve l'indice du k-ieme nombre"""
        if l == r:
            return l
        
        mid = (l + r) // 2
        
        # Compte dans left subtree
        left_count = 0
        if right_root and right_root.left:
            left_count += right_root.left.value
        if left_root and left_root.left:
            left_count -= left_root.left.value
        
        if k <= left_count:
            return self._kth(
                left_root.left if left_root else Node(),
                right_root.left if right_root else Node(),
                l, mid, k
            )
        else:
            return self._kth(
                left_root.right if left_root else Node(),
                right_root.right if right_root else Node(),
                mid + 1, r, k - left_count
            )


# ============================================================================
# TESTS UNITAIRES
# ============================================================================

def test_persistent_segment_tree_basic():
    """Test basic persistent segment tree"""
    pst = PersistentSegmentTree(5)
    
    v0 = pst.build([1, 2, 3, 4, 5])
    
    # Query v0
    assert pst.query(v0, 0, 4, 0, 4) == 15
    assert pst.query(v0, 0, 4, 1, 3) == 9
    
    # Update v1
    v1 = pst.update(v0, 0, 4, 2, 10)
    
    # v0 inchange
    assert pst.query(v0, 0, 4, 0, 4) == 15
    
    # v1 modifie
    assert pst.query(v1, 0, 4, 0, 4) == 22
    
    print("✓ Test persistent segment tree basic passed")


def test_persistent_segment_tree_multiple_versions():
    """Test multiple versions"""
    pst = PersistentSegmentTree(4)
    
    versions = [pst.build([0] * 4)]
    
    # Cree plusieurs versions
    versions.append(pst.update(versions[0], 0, 3, 0, 1))
    versions.append(pst.update(versions[1], 0, 3, 1, 2))
    versions.append(pst.update(versions[2], 0, 3, 2, 3))
    
    # Verifie chaque version
    assert pst.query(versions[0], 0, 3, 0, 3) == 0
    assert pst.query(versions[1], 0, 3, 0, 3) == 1
    assert pst.query(versions[2], 0, 3, 0, 3) == 3
    assert pst.query(versions[3], 0, 3, 0, 3) == 6
    
    print("✓ Test persistent segment tree multiple versions passed")


def test_persistent_kth_number():
    """Test kth number query"""
    arr = [3, 1, 4, 1, 5, 9, 2, 6]
    pst_kth = PersistentSegmentTreeKthNumber(arr)
    
    # Range [0, 4]: [3, 1, 4, 1, 5] -> sorted: [1, 1, 3, 4, 5]
    assert pst_kth.kth_number(0, 4, 1) == 1
    assert pst_kth.kth_number(0, 4, 3) == 3
    assert pst_kth.kth_number(0, 4, 5) == 5
    
    # Range [2, 5]: [4, 1, 5, 9] -> sorted: [1, 4, 5, 9]
    assert pst_kth.kth_number(2, 5, 2) == 4
    
    print("✓ Test persistent kth number passed")


# ============================================================================
# BENCHMARK
# ============================================================================

def benchmark_persistent_segment_tree():
    """Benchmark persistent segment tree"""
    import time
    import random
    
    print("\n=== Benchmark Persistent Segment Tree ===")
    
    for n in [1000, 5000, 10000]:
        pst = PersistentSegmentTree(n)
        
        arr = [random.randint(1, 100) for _ in range(n)]
        
        start = time.time()
        v = pst.build(arr)
        build_time = time.time() - start
        
        # Updates
        num_updates = 100
        versions = [v]
        
        start = time.time()
        for _ in range(num_updates):
            pos = random.randint(0, n-1)
            val = random.randint(1, 100)
            versions.append(pst.update(versions[-1], 0, n-1, pos, val))
        update_time = time.time() - start
        
        # Queries
        num_queries = 1000
        start = time.time()
        for _ in range(num_queries):
            ver = random.choice(versions)
            l = random.randint(0, n-1)
            r = random.randint(l, n-1)
            pst.query(ver, 0, n-1, l, r)
        query_time = time.time() - start
        
        print(f"\nn={n}:")
        print(f"  Build:   {build_time*1000:6.2f}ms")
        print(f"  Update:  {update_time/num_updates*1000:6.3f}ms/op")
        print(f"  Query:   {query_time/num_queries*1000:6.3f}ms/op")


if __name__ == "__main__":
    test_persistent_segment_tree_basic()
    test_persistent_segment_tree_multiple_versions()
    test_persistent_kth_number()
    
    benchmark_persistent_segment_tree()
    
    print("\n✓ Tous les tests Persistent Segment Tree passes!")

