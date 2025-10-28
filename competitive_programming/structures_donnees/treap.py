"""
================================================================================
TREAP (Tree + Heap)
================================================================================

Description:
-----------
Arbre binaire de recherche combine avec un tas (heap). Chaque noeud a une cle
(pour BST) et une priorite aleatoire (pour heap). Garantit O(log n) avec haute
probabilite. Supporte split/merge tres efficacement.

Complexite:
-----------
- Insert/Delete/Search: O(log n) attendu
- Split/Merge: O(log n) attendu
- Espace: O(n)

Cas d'usage typiques:
--------------------
1. Ensemble dynamique ordonne
2. Rope (chaine avec split/merge rapide)
3. Range operations avec lazy propagation
4. Implicit key treap (index comme cle)

Problemes classiques:
--------------------
- Codeforces 762E - Radio Stations
- SPOJ ORDERSET - Order Statistic Set
- Codeforces 675D - Tree Construction
- AtCoder ABC 134E - Sequence Decomposing

Auteur: Assistant CP
Date: 2025
================================================================================
"""

import random
from typing import Optional, Tuple, List


class Node:
    """Noeud du Treap"""
    
    def __init__(self, key: int):
        self.key = key
        self.priority = random.random()
        self.left: Optional[Node] = None
        self.right: Optional[Node] = None
        self.size = 1
    
    def update(self):
        """Met a jour la taille"""
        self.size = 1
        if self.left:
            self.size += self.left.size
        if self.right:
            self.size += self.right.size


class Treap:
    """
    Treap - arbre binaire de recherche randomise.
    
    Exemple:
    --------
    >>> treap = Treap()
    >>> for i in [5, 3, 7, 1, 9]:
    ...     treap.insert(i)
    >>> 
    >>> print(treap.contains(7))  # True
    >>> print(treap.contains(4))  # False
    >>> 
    >>> print(treap.kth(2))  # 3e plus petit = 5
    >>> 
    >>> treap.delete(5)
    >>> print(treap.contains(5))  # False
    """
    
    def __init__(self):
        """Initialise un treap vide"""
        self.root: Optional[Node] = None
    
    @staticmethod
    def _size(node: Optional[Node]) -> int:
        """Retourne la taille d'un noeud"""
        return node.size if node else 0
    
    def _split(self, node: Optional[Node], key: int) -> Tuple[Optional[Node], Optional[Node]]:
        """
        Split node en deux: left (< key) et right (>= key).
        
        Returns:
            (left, right)
        """
        if not node:
            return None, None
        
        if node.key < key:
            # node va dans left, split right
            left, right = self._split(node.right, key)
            node.right = left
            node.update()
            return node, right
        else:
            # node va dans right, split left
            left, right = self._split(node.left, key)
            node.left = right
            node.update()
            return left, node
    
    def _merge(self, left: Optional[Node], right: Optional[Node]) -> Optional[Node]:
        """
        Merge deux treaps.
        Prerequis: toutes les cles de left < toutes les cles de right
        
        Returns:
            Racine du treap merge
        """
        if not left:
            return right
        if not right:
            return left
        
        if left.priority > right.priority:
            # left devient racine
            left.right = self._merge(left.right, right)
            left.update()
            return left
        else:
            # right devient racine
            right.left = self._merge(left, right.left)
            right.update()
            return right
    
    def insert(self, key: int):
        """
        Insere une cle.
        
        Time: O(log n) attendu
        """
        left, right = self._split(self.root, key)
        middle, right = self._split(right, key + 1)
        
        if not middle:
            middle = Node(key)
        
        self.root = self._merge(self._merge(left, middle), right)
    
    def delete(self, key: int) -> bool:
        """
        Supprime une cle.
        
        Time: O(log n) attendu
        
        Returns:
            True si la cle etait presente
        """
        left, right = self._split(self.root, key)
        middle, right = self._split(right, key + 1)
        
        self.root = self._merge(left, right)
        return middle is not None
    
    def contains(self, key: int) -> bool:
        """
        Verifie si la cle existe.
        
        Time: O(log n) attendu
        """
        node = self.root
        while node:
            if key == node.key:
                return True
            elif key < node.key:
                node = node.left
            else:
                node = node.right
        return False
    
    def size(self) -> int:
        """Retourne le nombre d'elements"""
        return self._size(self.root)
    
    def kth(self, k: int) -> Optional[int]:
        """
        Trouve le k-ieme plus petit element (0-indexed).
        
        Time: O(log n) attendu
        
        Args:
            k: Index (0 = min, size-1 = max)
            
        Returns:
            La k-ieme cle ou None si k invalide
        """
        if k < 0 or k >= self.size():
            return None
        
        node = self.root
        while node:
            left_size = self._size(node.left)
            
            if k == left_size:
                return node.key
            elif k < left_size:
                node = node.left
            else:
                k -= left_size + 1
                node = node.right
        
        return None
    
    def split(self, key: int) -> Tuple['Treap', 'Treap']:
        """
        Split en deux treaps: left (< key) et right (>= key).
        
        Time: O(log n) attendu
        
        Args:
            key: Valeur de split
            
        Returns:
            (left_treap, right_treap)
        """
        left_node, right_node = self._split(self.root, key)
        
        left_treap = Treap()
        left_treap.root = left_node
        
        right_treap = Treap()
        right_treap.root = right_node
        
        self.root = None
        
        return left_treap, right_treap
    
    def merge(self, other: 'Treap'):
        """
        Merge avec un autre treap.
        Prerequis: toutes les cles de self < toutes les cles de other
        
        Time: O(log n) attendu
        """
        self.root = self._merge(self.root, other.root)
        other.root = None
    
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


class ImplicitTreap:
    """
    Treap avec cles implicites (index).
    Utile pour sequences avec operations de range.
    
    Exemple:
    --------
    >>> it = ImplicitTreap([1, 2, 3, 4, 5])
    >>> 
    >>> # Reverse range [1, 3]
    >>> it.reverse(1, 3)
    >>> print(it.to_list())  # [1, 4, 3, 2, 5]
    >>> 
    >>> # Insert element
    >>> it.insert(2, 10)
    >>> print(it.to_list())  # [1, 4, 10, 3, 2, 5]
    """
    
    class Node:
        def __init__(self, value: int):
            self.value = value
            self.priority = random.random()
            self.left: Optional['ImplicitTreap.Node'] = None
            self.right: Optional['ImplicitTreap.Node'] = None
            self.size = 1
            self.reversed = False
        
        def push(self):
            """Propage lazy tag"""
            if self.reversed:
                self.left, self.right = self.right, self.left
                if self.left:
                    self.left.reversed ^= True
                if self.right:
                    self.right.reversed ^= True
                self.reversed = False
        
        def update(self):
            """Met a jour size"""
            self.size = 1
            if self.left:
                self.size += self.left.size
            if self.right:
                self.size += self.right.size
    
    def __init__(self, arr: List[int] = None):
        """Initialise depuis un tableau"""
        self.root: Optional[ImplicitTreap.Node] = None
        if arr:
            for val in arr:
                self.push_back(val)
    
    @staticmethod
    def _size(node: Optional['ImplicitTreap.Node']) -> int:
        return node.size if node else 0
    
    def _split(self, node: Optional['ImplicitTreap.Node'], 
               pos: int) -> Tuple[Optional['ImplicitTreap.Node'], Optional['ImplicitTreap.Node']]:
        """Split par position"""
        if not node:
            return None, None
        
        node.push()
        left_size = self._size(node.left)
        
        if left_size < pos:
            left, right = self._split(node.right, pos - left_size - 1)
            node.right = left
            node.update()
            return node, right
        else:
            left, right = self._split(node.left, pos)
            node.left = right
            node.update()
            return left, node
    
    def _merge(self, left: Optional['ImplicitTreap.Node'], 
               right: Optional['ImplicitTreap.Node']) -> Optional['ImplicitTreap.Node']:
        """Merge deux treaps"""
        if not left:
            return right
        if not right:
            return left
        
        left.push()
        right.push()
        
        if left.priority > right.priority:
            left.right = self._merge(left.right, right)
            left.update()
            return left
        else:
            right.left = self._merge(left, right.left)
            right.update()
            return right
    
    def insert(self, pos: int, value: int):
        """Insere un element a la position pos"""
        left, right = self._split(self.root, pos)
        new_node = ImplicitTreap.Node(value)
        self.root = self._merge(self._merge(left, new_node), right)
    
    def erase(self, pos: int):
        """Supprime l'element a la position pos"""
        left, right = self._split(self.root, pos)
        middle, right = self._split(right, 1)
        self.root = self._merge(left, right)
    
    def push_back(self, value: int):
        """Ajoute a la fin"""
        new_node = ImplicitTreap.Node(value)
        self.root = self._merge(self.root, new_node)
    
    def reverse(self, l: int, r: int):
        """Reverse le range [l, r)"""
        left, mid_right = self._split(self.root, l)
        mid, right = self._split(mid_right, r - l)
        
        if mid:
            mid.reversed ^= True
        
        self.root = self._merge(self._merge(left, mid), right)
    
    def size(self) -> int:
        return self._size(self.root)
    
    def to_list(self) -> List[int]:
        """Retourne le tableau"""
        result = []
        
        def inorder(node: Optional['ImplicitTreap.Node']):
            if not node:
                return
            node.push()
            inorder(node.left)
            result.append(node.value)
            inorder(node.right)
        
        inorder(self.root)
        return result


# ============================================================================
# TESTS UNITAIRES
# ============================================================================

def test_treap_basic():
    """Test basic treap operations"""
    treap = Treap()
    
    for i in [5, 3, 7, 1, 9]:
        treap.insert(i)
    
    assert treap.contains(5)
    assert treap.contains(1)
    assert not treap.contains(4)
    assert treap.size() == 5
    
    print("✓ Test treap basic passed")


def test_treap_delete():
    """Test delete"""
    treap = Treap()
    
    for i in [5, 3, 7, 1, 9]:
        treap.insert(i)
    
    assert treap.delete(5)
    assert not treap.contains(5)
    assert treap.size() == 4
    
    print("✓ Test treap delete passed")


def test_treap_kth():
    """Test kth element"""
    treap = Treap()
    
    for i in [5, 3, 7, 1, 9]:
        treap.insert(i)
    
    assert treap.kth(0) == 1
    assert treap.kth(2) == 5
    assert treap.kth(4) == 9
    
    print("✓ Test treap kth passed")


def test_treap_split_merge():
    """Test split and merge"""
    treap = Treap()
    
    for i in [1, 3, 5, 7, 9]:
        treap.insert(i)
    
    left, right = treap.split(5)
    
    assert sorted(left.to_list()) == [1, 3]
    assert sorted(right.to_list()) == [5, 7, 9]
    
    left.merge(right)
    assert sorted(left.to_list()) == [1, 3, 5, 7, 9]
    
    print("✓ Test treap split/merge passed")


def test_implicit_treap():
    """Test implicit treap"""
    it = ImplicitTreap([1, 2, 3, 4, 5])
    
    assert it.to_list() == [1, 2, 3, 4, 5]
    
    # Insert
    it.insert(2, 10)
    assert it.to_list() == [1, 2, 10, 3, 4, 5]
    
    # Erase
    it.erase(2)
    assert it.to_list() == [1, 2, 3, 4, 5]
    
    print("✓ Test implicit treap passed")


def test_implicit_treap_reverse():
    """Test reverse operation"""
    it = ImplicitTreap([1, 2, 3, 4, 5])
    
    # Reverse [1, 4)
    it.reverse(1, 4)
    assert it.to_list() == [1, 4, 3, 2, 5]
    
    # Reverse all
    it.reverse(0, 5)
    assert it.to_list() == [5, 2, 3, 4, 1]
    
    print("✓ Test implicit treap reverse passed")


# ============================================================================
# BENCHMARK
# ============================================================================

def benchmark_treap():
    """Benchmark treap"""
    import time
    
    print("\n=== Benchmark Treap ===")
    
    for n in [1000, 5000, 10000]:
        treap = Treap()
        
        values = list(range(n))
        random.shuffle(values)
        
        start = time.time()
        for v in values:
            treap.insert(v)
        insert_time = time.time() - start
        
        random.shuffle(values)
        start = time.time()
        for v in values[:1000]:
            treap.contains(v)
        search_time = time.time() - start
        
        print(f"\nn={n}:")
        print(f"  Insert: {insert_time*1000:6.2f}ms")
        print(f"  Search: {search_time:6.3f}ms (1000 queries)")


if __name__ == "__main__":
    # Tests
    test_treap_basic()
    test_treap_delete()
    test_treap_kth()
    test_treap_split_merge()
    test_implicit_treap()
    test_implicit_treap_reverse()
    
    # Benchmark
    benchmark_treap()
    
    print("\n✓ Tous les tests Treap passes!")

