"""
================================================================================
SPLAY TREE
================================================================================

Description:
-----------
Arbre binaire de recherche auto-equilibrant qui reorganise l'arbre a chaque
acces en faisant "splay" (remonter) le noeud accede jusqu'a la racine.
Garantit O(log n) amorti pour toutes les operations.

Complexite:
-----------
- Insert/Delete/Search: O(log n) amorti
- Split/Merge: O(log n) amorti
- Espace: O(n)

Cas d'usage typiques:
--------------------
1. Maintenir un ensemble dynamique ordonne
2. Split/Merge efficaces
3. Range queries avec lazy propagation
4. Cache-friendly (elements recents en haut)

Problemes classiques:
--------------------
- Codeforces 762E - Radio Stations
- SPOJ ORDERSET - Order Statistic Set
- Codeforces 675D - Tree Construction
- AtCoder ABC 150E - Change a Little Bit

Auteur: Assistant CP
Date: 2025
================================================================================
"""

from typing import Optional, Tuple, List


class Node:
    """Noeud du Splay Tree"""
    
    def __init__(self, key: int):
        self.key = key
        self.left: Optional[Node] = None
        self.right: Optional[Node] = None
        self.parent: Optional[Node] = None
        self.size = 1  # Taille du sous-arbre
    
    def update(self):
        """Met a jour la taille du sous-arbre"""
        self.size = 1
        if self.left:
            self.size += self.left.size
        if self.right:
            self.size += self.right.size


class SplayTree:
    """
    Splay Tree - arbre binaire de recherche auto-equilibrant.
    
    Exemple:
    --------
    >>> st = SplayTree()
    >>> for i in [5, 3, 7, 1, 9]:
    ...     st.insert(i)
    >>> 
    >>> print(st.contains(7))  # True
    >>> print(st.contains(4))  # False
    >>> 
    >>> print(st.kth(2))  # 3e plus petit = 5
    >>> 
    >>> st.delete(5)
    >>> print(st.contains(5))  # False
    """
    
    def __init__(self):
        """Initialise un splay tree vide"""
        self.root: Optional[Node] = None
    
    def _set_parent(self, child: Optional[Node], parent: Optional[Node]):
        """Definit le parent d'un noeud"""
        if child:
            child.parent = parent
    
    def _rotate_right(self, node: Node):
        """Rotation droite autour de node"""
        left = node.left
        node.left = left.right
        self._set_parent(node.left, node)
        left.right = node
        
        parent = node.parent
        self._set_parent(left, parent)
        
        if parent:
            if parent.left == node:
                parent.left = left
            else:
                parent.right = left
        
        self._set_parent(node, left)
        
        node.update()
        left.update()
        
        return left
    
    def _rotate_left(self, node: Node):
        """Rotation gauche autour de node"""
        right = node.right
        node.right = right.left
        self._set_parent(node.right, node)
        right.left = node
        
        parent = node.parent
        self._set_parent(right, parent)
        
        if parent:
            if parent.left == node:
                parent.left = right
            else:
                parent.right = right
        
        self._set_parent(node, right)
        
        node.update()
        right.update()
        
        return right
    
    def _splay(self, node: Node):
        """Remonte node jusqu'a la racine avec rotations"""
        while node.parent:
            parent = node.parent
            grand = parent.parent
            
            if not grand:
                # Zig
                if parent.left == node:
                    self._rotate_right(parent)
                else:
                    self._rotate_left(parent)
            elif grand.left == parent and parent.left == node:
                # Zig-zig (gauche-gauche)
                self._rotate_right(grand)
                self._rotate_right(parent)
            elif grand.right == parent and parent.right == node:
                # Zig-zig (droite-droite)
                self._rotate_left(grand)
                self._rotate_left(parent)
            elif grand.left == parent and parent.right == node:
                # Zig-zag (gauche-droite)
                self._rotate_left(parent)
                self._rotate_right(grand)
            else:
                # Zig-zag (droite-gauche)
                self._rotate_right(parent)
                self._rotate_left(grand)
        
        self.root = node
    
    def _find(self, key: int) -> Optional[Node]:
        """Trouve le noeud avec la cle donnee"""
        node = self.root
        last = None
        
        while node:
            last = node
            if key == node.key:
                self._splay(node)
                return node
            elif key < node.key:
                node = node.left
            else:
                node = node.right
        
        if last:
            self._splay(last)
        
        return None
    
    def contains(self, key: int) -> bool:
        """
        Verifie si la cle existe.
        
        Time: O(log n) amorti
        """
        return self._find(key) is not None
    
    def insert(self, key: int):
        """
        Insere une cle.
        
        Time: O(log n) amorti
        """
        if not self.root:
            self.root = Node(key)
            return
        
        node = self.root
        while True:
            if key == node.key:
                # Deja present
                self._splay(node)
                return
            elif key < node.key:
                if not node.left:
                    node.left = Node(key)
                    self._set_parent(node.left, node)
                    node.update()
                    self._splay(node.left)
                    return
                node = node.left
            else:
                if not node.right:
                    node.right = Node(key)
                    self._set_parent(node.right, node)
                    node.update()
                    self._splay(node.right)
                    return
                node = node.right
    
    def delete(self, key: int) -> bool:
        """
        Supprime une cle.
        
        Time: O(log n) amorti
        
        Returns:
            True si la cle etait presente
        """
        node = self._find(key)
        if not node:
            return False
        
        # Maintenant node est la racine
        if not node.left:
            self.root = node.right
            self._set_parent(self.root, None)
        elif not node.right:
            self.root = node.left
            self._set_parent(self.root, None)
        else:
            # Trouve le successeur (min du sous-arbre droit)
            succ = node.right
            while succ.left:
                succ = succ.left
            
            # Splay le successeur
            self._splay(succ)
            
            # succ devient la racine et prend le left de node
            succ.left = node.left
            self._set_parent(succ.left, succ)
            succ.update()
        
        return True
    
    def size(self) -> int:
        """Retourne le nombre d'elements"""
        return self.root.size if self.root else 0
    
    def kth(self, k: int) -> Optional[int]:
        """
        Trouve le k-ieme plus petit element (0-indexed).
        
        Time: O(log n) amorti
        
        Args:
            k: Index (0 = min, size-1 = max)
            
        Returns:
            La k-ieme cle ou None si k invalide
        """
        if k < 0 or k >= self.size():
            return None
        
        node = self.root
        while node:
            left_size = node.left.size if node.left else 0
            
            if k == left_size:
                self._splay(node)
                return node.key
            elif k < left_size:
                node = node.left
            else:
                k -= left_size + 1
                node = node.right
        
        return None
    
    def split(self, key: int) -> Tuple['SplayTree', 'SplayTree']:
        """
        Split en deux arbres: left (< key) et right (>= key).
        
        Time: O(log n) amorti
        
        Args:
            key: Valeur de split
            
        Returns:
            (left_tree, right_tree)
        """
        if not self.root:
            return SplayTree(), SplayTree()
        
        # Trouve le noeud >= key
        node = self.root
        last = None
        
        while node:
            last = node
            if key <= node.key:
                node = node.left
            else:
                node = node.right
        
        self._splay(last)
        
        left_tree = SplayTree()
        right_tree = SplayTree()
        
        if self.root.key < key:
            # root va dans left, right va dans right
            left_tree.root = self.root
            right_tree.root = self.root.right
            if left_tree.root:
                left_tree.root.right = None
                self._set_parent(right_tree.root, None)
                left_tree.root.update()
        else:
            # left va dans left, root va dans right
            right_tree.root = self.root
            left_tree.root = self.root.left
            if right_tree.root:
                right_tree.root.left = None
                self._set_parent(left_tree.root, None)
                right_tree.root.update()
        
        return left_tree, right_tree
    
    def merge(self, other: 'SplayTree'):
        """
        Merge avec un autre splay tree.
        Prerequis: toutes les cles de self < toutes les cles de other
        
        Time: O(log n) amorti
        """
        if not self.root:
            self.root = other.root
            return
        
        if not other.root:
            return
        
        # Trouve le max de self
        node = self.root
        while node.right:
            node = node.right
        
        self._splay(node)
        
        # Attache other a droite
        self.root.right = other.root
        self._set_parent(other.root, self.root)
        self.root.update()
    
    def to_list(self) -> List[int]:
        """Retourne les elements en ordre trie"""
        result = []
        
        def inorder(node: Optional[Node]):
            if not node:
                return
            inorder(node.left)
            result.append(node.key)
            inorder(node.right)
        
        inorder(self.root)
        return result


# ============================================================================
# TESTS UNITAIRES
# ============================================================================

def test_splay_tree_basic():
    """Test basic splay tree operations"""
    st = SplayTree()
    
    # Insert
    for i in [5, 3, 7, 1, 9]:
        st.insert(i)
    
    # Contains
    assert st.contains(5)
    assert st.contains(1)
    assert not st.contains(4)
    
    # Size
    assert st.size() == 5
    
    print("✓ Test splay tree basic passed")


def test_splay_tree_delete():
    """Test delete operation"""
    st = SplayTree()
    
    for i in [5, 3, 7, 1, 9]:
        st.insert(i)
    
    assert st.delete(5)
    assert not st.contains(5)
    assert st.size() == 4
    
    assert not st.delete(10)
    assert st.size() == 4
    
    print("✓ Test splay tree delete passed")


def test_splay_tree_kth():
    """Test kth element"""
    st = SplayTree()
    
    for i in [5, 3, 7, 1, 9]:
        st.insert(i)
    
    # Ordre: 1, 3, 5, 7, 9
    assert st.kth(0) == 1
    assert st.kth(1) == 3
    assert st.kth(2) == 5
    assert st.kth(3) == 7
    assert st.kth(4) == 9
    
    assert st.kth(5) is None
    
    print("✓ Test splay tree kth passed")


def test_splay_tree_split():
    """Test split operation"""
    st = SplayTree()
    
    for i in [1, 3, 5, 7, 9]:
        st.insert(i)
    
    left, right = st.split(5)
    
    # left: {1, 3}, right: {5, 7, 9}
    assert sorted(left.to_list()) == [1, 3]
    assert sorted(right.to_list()) == [5, 7, 9]
    
    print("✓ Test splay tree split passed")


def test_splay_tree_merge():
    """Test merge operation"""
    st1 = SplayTree()
    for i in [1, 3]:
        st1.insert(i)
    
    st2 = SplayTree()
    for i in [5, 7, 9]:
        st2.insert(i)
    
    st1.merge(st2)
    
    assert sorted(st1.to_list()) == [1, 3, 5, 7, 9]
    
    print("✓ Test splay tree merge passed")


def test_splay_tree_duplicates():
    """Test avec doublons"""
    st = SplayTree()
    
    st.insert(5)
    st.insert(5)
    st.insert(5)
    
    assert st.size() == 1  # Pas de doublons
    assert st.to_list() == [5]
    
    print("✓ Test splay tree duplicates passed")


# ============================================================================
# BENCHMARK
# ============================================================================

def benchmark_splay_tree():
    """Benchmark splay tree"""
    import time
    import random
    
    print("\n=== Benchmark Splay Tree ===")
    
    for n in [1000, 5000, 10000]:
        st = SplayTree()
        
        # Insert
        values = list(range(n))
        random.shuffle(values)
        
        start = time.time()
        for v in values:
            st.insert(v)
        insert_time = time.time() - start
        
        # Search
        random.shuffle(values)
        start = time.time()
        for v in values[:1000]:
            st.contains(v)
        search_time = time.time() - start
        
        # Kth
        start = time.time()
        for _ in range(1000):
            st.kth(random.randint(0, n-1))
        kth_time = time.time() - start
        
        print(f"\nn={n}:")
        print(f"  Insert: {insert_time*1000:6.2f}ms")
        print(f"  Search: {search_time:6.3f}ms (1000 queries)")
        print(f"  Kth:    {kth_time:6.3f}ms (1000 queries)")


if __name__ == "__main__":
    # Tests
    test_splay_tree_basic()
    test_splay_tree_delete()
    test_splay_tree_kth()
    test_splay_tree_split()
    test_splay_tree_merge()
    test_splay_tree_duplicates()
    
    # Benchmark
    benchmark_splay_tree()
    
    print("\n✓ Tous les tests Splay Tree passes!")

