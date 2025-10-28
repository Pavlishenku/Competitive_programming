"""
================================================================================
WAVELET TREE
================================================================================

Description:
-----------
Structure hierarchique pour repondre a des queries sur sequences.
Supporte k-th smallest, count occurrences, range queries en O(log alphabet).

Complexite:
-----------
- Construction: O(n log alphabet)
- Kth smallest: O(log alphabet)
- Count in range: O(log alphabet)
- Espace: O(n log alphabet)

Cas d'usage typiques:
--------------------
1. K-th smallest in range
2. Count occurrences in range
3. Quantile queries
4. 2D range queries

Problemes classiques:
--------------------
- SPOJ MKTHNUM - K-th Number
- Codeforces 840D - Destiny
- CSES Range Queries
- AtCoder ABC 172F - Unfair Nim

Auteur: Assistant CP
Date: 2025
================================================================================
"""

from typing import List, Optional


class WaveletTree:
    """
    Wavelet Tree pour queries sur sequences.
    
    Exemple:
    --------
    >>> arr = [3, 1, 4, 1, 5, 9, 2, 6, 5]
    >>> wt = WaveletTree(arr)
    >>> 
    >>> # Kth smallest in range [2, 6]
    >>> print(wt.kth_smallest(2, 6, 2))  # 2e plus petit = 2
    >>> 
    >>> # Count occurrences of 5 in [0, 8]
    >>> print(wt.count_range(0, 8, 5, 5))  # 2
    """
    
    def __init__(self, arr: List[int]):
        """
        Args:
            arr: Tableau d'entree
        """
        self.arr = arr
        self.n = len(arr)
        
        if self.n == 0:
            return
        
        # Compression des valeurs
        self.sorted_vals = sorted(set(arr))
        self.val_to_idx = {v: i for i, v in enumerate(self.sorted_vals)}
        
        # Construit l'arbre
        compressed = [self.val_to_idx[x] for x in arr]
        self.tree = {}
        self._build(compressed, 0, len(self.sorted_vals) - 1, 0, self.n - 1)
    
    def _build(self, arr: List[int], val_l: int, val_r: int, pos_l: int, pos_r: int):
        """Construction recursive"""
        if pos_l > pos_r or val_l > val_r:
            return
        
        if val_l == val_r:
            return
        
        val_mid = (val_l + val_r) // 2
        
        # Bitmap: 0 si <= val_mid, 1 sinon
        bitmap = []
        left_arr = []
        right_arr = []
        
        for i in range(pos_l, pos_r + 1):
            if arr[i] <= val_mid:
                bitmap.append(0)
                left_arr.append(arr[i])
            else:
                bitmap.append(1)
                right_arr.append(arr[i])
        
        # Prefix sum pour bitmap
        prefix = [0]
        for b in bitmap:
            prefix.append(prefix[-1] + b)
        
        self.tree[(val_l, val_r, pos_l)] = (bitmap, prefix)
        
        # Recurse
        if left_arr:
            self._build(left_arr, val_l, val_mid, pos_l, pos_l + len(left_arr) - 1)
        if right_arr:
            self._build(right_arr, val_mid + 1, val_r, 
                       pos_l + len(left_arr), pos_r)
    
    def _rank(self, bitmap: List[int], prefix: List[int], pos: int, bit: int) -> int:
        """Compte le nombre de 'bit' dans bitmap[0:pos+1]"""
        if pos < 0:
            return 0
        if bit == 1:
            return prefix[pos + 1]
        else:
            return (pos + 1) - prefix[pos + 1]
    
    def kth_smallest(self, l: int, r: int, k: int) -> Optional[int]:
        """
        Trouve le k-ieme plus petit element dans arr[l:r+1].
        
        Time: O(log alphabet)
        
        Args:
            l, r: Bornes du range (inclusive)
            k: Position (1-indexed)
            
        Returns:
            Le k-ieme plus petit element
        """
        if k < 1 or k > r - l + 1:
            return None
        
        return self.sorted_vals[
            self._kth_smallest(0, len(self.sorted_vals) - 1, 0, l, r, k)
        ]
    
    def _kth_smallest(self, val_l: int, val_r: int, pos_l: int, 
                     l: int, r: int, k: int) -> int:
        """Helper recursif"""
        if val_l == val_r:
            return val_l
        
        key = (val_l, val_r, pos_l)
        if key not in self.tree:
            return val_l
        
        bitmap, prefix = self.tree[key]
        
        val_mid = (val_l + val_r) // 2
        
        # Compte zeros dans [l, r]
        rank_l_0 = self._rank(bitmap, prefix, l - pos_l - 1, 0)
        rank_r_0 = self._rank(bitmap, prefix, r - pos_l, 0)
        zeros_count = rank_r_0 - rank_l_0
        
        if k <= zeros_count:
            # Kth est dans left
            new_l = pos_l + rank_l_0
            new_r = pos_l + rank_r_0 - 1
            return self._kth_smallest(val_l, val_mid, pos_l, new_l, new_r, k)
        else:
            # Kth est dans right
            rank_l_1 = self._rank(bitmap, prefix, l - pos_l - 1, 1)
            rank_r_1 = self._rank(bitmap, prefix, r - pos_l, 1)
            
            left_size = rank_r_0 - rank_l_0
            new_pos_l = pos_l + (r - l + 1) - (rank_r_1 - rank_l_1)
            
            new_l = new_pos_l + rank_l_1
            new_r = new_pos_l + rank_r_1 - 1
            
            return self._kth_smallest(val_mid + 1, val_r, new_pos_l, 
                                     new_l, new_r, k - zeros_count)
    
    def count_range(self, l: int, r: int, val_l: int, val_r: int) -> int:
        """
        Compte le nombre d'elements dans arr[l:r+1] avec valeur dans [val_l, val_r].
        
        Time: O(log alphabet)
        
        Args:
            l, r: Bornes du range de positions
            val_l, val_r: Bornes du range de valeurs
            
        Returns:
            Nombre d'elements
        """
        if val_l not in self.val_to_idx or val_r not in self.val_to_idx:
            # Ajuste aux valeurs presentes
            val_l_idx = 0
            val_r_idx = len(self.sorted_vals) - 1
            
            for v in self.sorted_vals:
                if v >= val_l:
                    val_l_idx = self.val_to_idx[v]
                    break
            
            for v in reversed(self.sorted_vals):
                if v <= val_r:
                    val_r_idx = self.val_to_idx[v]
                    break
        else:
            val_l_idx = self.val_to_idx[val_l]
            val_r_idx = self.val_to_idx[val_r]
        
        if val_l_idx > val_r_idx:
            return 0
        
        return self._count_range(0, len(self.sorted_vals) - 1, 0, 
                                l, r, val_l_idx, val_r_idx)
    
    def _count_range(self, val_l: int, val_r: int, pos_l: int, 
                    l: int, r: int, qval_l: int, qval_r: int) -> int:
        """Helper recursif"""
        if qval_l > val_r or qval_r < val_l or l > r:
            return 0
        
        if qval_l <= val_l and val_r <= qval_r:
            return r - l + 1
        
        key = (val_l, val_r, pos_l)
        if key not in self.tree:
            return 0
        
        bitmap, prefix = self.tree[key]
        val_mid = (val_l + val_r) // 2
        
        rank_l_0 = self._rank(bitmap, prefix, l - pos_l - 1, 0)
        rank_r_0 = self._rank(bitmap, prefix, r - pos_l, 0)
        
        rank_l_1 = self._rank(bitmap, prefix, l - pos_l - 1, 1)
        rank_r_1 = self._rank(bitmap, prefix, r - pos_l, 1)
        
        result = 0
        
        # Left subtree
        if qval_l <= val_mid:
            new_l = pos_l + rank_l_0
            new_r = pos_l + rank_r_0 - 1
            if new_l <= new_r:
                result += self._count_range(val_l, val_mid, pos_l, 
                                           new_l, new_r, qval_l, qval_r)
        
        # Right subtree
        if qval_r > val_mid:
            left_size = rank_r_0 - rank_l_0
            new_pos_l = pos_l + left_size
            new_l = new_pos_l + rank_l_1
            new_r = new_pos_l + rank_r_1 - 1
            if new_l <= new_r:
                result += self._count_range(val_mid + 1, val_r, new_pos_l, 
                                           new_l, new_r, qval_l, qval_r)
        
        return result


# ============================================================================
# TESTS UNITAIRES
# ============================================================================

def test_wavelet_tree_kth():
    """Test kth smallest"""
    arr = [3, 1, 4, 1, 5, 9, 2, 6, 5]
    wt = WaveletTree(arr)
    
    # Range [0, 8]: sorted = [1, 1, 2, 3, 4, 5, 5, 6, 9]
    assert wt.kth_smallest(0, 8, 1) == 1
    assert wt.kth_smallest(0, 8, 5) == 4
    assert wt.kth_smallest(0, 8, 9) == 9
    
    # Range [2, 6]: [4, 1, 5, 9, 2] sorted = [1, 2, 4, 5, 9]
    assert wt.kth_smallest(2, 6, 2) == 2
    
    print("✓ Test wavelet tree kth passed")


def test_wavelet_tree_count():
    """Test count range"""
    arr = [3, 1, 4, 1, 5, 9, 2, 6, 5]
    wt = WaveletTree(arr)
    
    # Count 5 in [0, 8]
    assert wt.count_range(0, 8, 5, 5) == 2
    
    # Count values in [1, 4] in range [0, 8]
    assert wt.count_range(0, 8, 1, 4) == 5  # 1,1,3,4,2
    
    print("✓ Test wavelet tree count passed")


# ============================================================================
# BENCHMARK
# ============================================================================

def benchmark_wavelet_tree():
    """Benchmark wavelet tree"""
    import time
    import random
    
    print("\n=== Benchmark Wavelet Tree ===")
    
    for n in [1000, 5000, 10000]:
        arr = [random.randint(1, 100) for _ in range(n)]
        
        start = time.time()
        wt = WaveletTree(arr)
        build_time = time.time() - start
        
        # Kth queries
        num_queries = 1000
        start = time.time()
        for _ in range(num_queries):
            l = random.randint(0, n-1)
            r = random.randint(l, n-1)
            k = random.randint(1, r - l + 1)
            wt.kth_smallest(l, r, k)
        kth_time = time.time() - start
        
        print(f"\nn={n}:")
        print(f"  Build: {build_time*1000:6.2f}ms")
        print(f"  Kth:   {kth_time/num_queries*1000:6.3f}ms/query")


if __name__ == "__main__":
    test_wavelet_tree_kth()
    test_wavelet_tree_count()
    
    benchmark_wavelet_tree()
    
    print("\n✓ Tous les tests Wavelet Tree passes!")

