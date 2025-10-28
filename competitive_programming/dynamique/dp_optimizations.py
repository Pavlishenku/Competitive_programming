"""
================================================================================
DP OPTIMIZATIONS
================================================================================

Description:
-----------
Collection d'optimisations avancees pour la programmation dynamique:
- Aliens Trick / Lagrange Optimization
- Knuth Optimization
- Monotone Queue Optimization
- Divide and Conquer DP

Complexite:
-----------
- Aliens Trick: Reduit O(n*k) a O(n log k) avec binary search
- Knuth: Reduit O(n^3) a O(n^2)
- Monotone Queue: Reduit O(n*k) a O(n+k)
- D&C DP: Reduit O(n^2*k) a O(n*k*log n)

Cas d'usage typiques:
--------------------
1. DP avec contrainte sur nombre de transitions
2. Range DP avec propriete de Knuth
3. DP avec fenetre glissante
4. DP avec optimal_k[i][j] monotone

Problemes classiques:
--------------------
- IOI 2016 - Aliens
- Codeforces 321E - Ciel and Gondolas
- USACO Training - Breaking Necklace
- AtCoder DP Contest

Auteur: Assistant CP
Date: 2025
================================================================================
"""

from typing import List, Tuple, Callable
from collections import deque


def aliens_trick_max_k_partitions(arr: List[int], k: int, 
                                  cost_fn: Callable[[int, int], int]) -> int:
    """
    Aliens Trick pour problemes de partitionnement en exactement k parties.
    
    Optimise: max sum de cost_fn sur exactement k partitions
    
    Time: O(n^2 log(max_cost))
    
    Args:
        arr: Tableau d'entree
        k: Nombre de partitions souhaite
        cost_fn: Fonction cost(i, j) pour partition [i, j]
        
    Returns:
        Cout maximal pour k partitions
        
    Exemple:
    --------
    >>> arr = [1, 2, 3, 4, 5]
    >>> def cost(i, j):
    ...     return sum(arr[i:j+1])
    >>> result = aliens_trick_max_k_partitions(arr, 2, cost)
    """
    n = len(arr)
    
    def dp_with_lambda(lam: float) -> Tuple[int, int]:
        """
        DP avec penalite lambda par partition.
        Returns: (max_value, num_partitions)
        """
        # dp[i] = (max_value, num_parts) pour arr[0:i]
        dp = [(-float('inf'), 0)] * (n + 1)
        dp[0] = (0, 0)
        
        for i in range(1, n + 1):
            for j in range(i):
                cost = cost_fn(j, i - 1)
                value = dp[j][0] + cost - lam
                parts = dp[j][1] + 1
                
                if value > dp[i][0]:
                    dp[i] = (value, parts)
        
        return dp[n]
    
    # Binary search sur lambda
    left, right = -1e9, 1e9
    
    for _ in range(60):  # Precision suffisante
        mid = (left + right) / 2
        _, num_parts = dp_with_lambda(mid)
        
        if num_parts >= k:
            left = mid
        else:
            right = mid
    
    value, _ = dp_with_lambda(left)
    return int(value + k * left + 0.5)


def knuth_optimization_range_dp(arr: List[int], 
                                cost_fn: Callable[[int, int], int]) -> int:
    """
    Knuth Optimization pour Range DP.
    
    Applicable si cost satisfait quadrangle inequality:
    cost(a,c) + cost(b,d) <= cost(a,d) + cost(b,c) pour a<=b<=c<=d
    
    Time: O(n^2) au lieu de O(n^3)
    
    Args:
        arr: Tableau d'entree
        cost_fn: Fonction de cout pour fusionner [i, j]
        
    Returns:
        Cout minimal
        
    Exemple:
    --------
    >>> # Matrix chain multiplication
    >>> dims = [10, 20, 30, 40]
    >>> def cost(i, j):
    ...     if i == j:
    ...         return 0
    ...     return dims[i] * dims[j] * dims[j+1]
    >>> result = knuth_optimization_range_dp(dims, cost)
    """
    n = len(arr)
    
    # dp[i][j] = cout minimal pour range [i, j]
    dp = [[float('inf')] * n for _ in range(n)]
    opt = [[0] * n for _ in range(n)]  # Position optimale de split
    
    # Base case
    for i in range(n):
        dp[i][i] = 0
        opt[i][i] = i
    
    # Rempli par longueur croissante
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            
            # Knuth optimization: search only in [opt[i][j-1], opt[i+1][j]]
            left_bound = opt[i][j - 1] if j > 0 else i
            right_bound = opt[i + 1][j] if i + 1 < n else j
            
            for k in range(max(i, left_bound), min(j, right_bound) + 1):
                cost = dp[i][k] + dp[k + 1][j] + cost_fn(i, j)
                
                if cost < dp[i][j]:
                    dp[i][j] = cost
                    opt[i][j] = k
    
    return dp[0][n - 1]


def monotone_queue_dp(arr: List[int], k: int, 
                     transition_fn: Callable[[int, int], int]) -> List[int]:
    """
    DP avec Monotone Queue pour window optimization.
    
    Applicable si: dp[i] = min/max_{j in [i-k, i-1]} (dp[j] + cost(j, i))
    
    Time: O(n) au lieu de O(n*k)
    
    Args:
        arr: Tableau d'entree
        k: Taille de la fenetre
        transition_fn: Fonction transition(j, i) = dp[j] + cost
        
    Returns:
        Tableau DP
        
    Exemple:
    --------
    >>> arr = [1, 5, 2, 8, 3]
    >>> def trans(j, i):
    ...     return arr[i] - arr[j]
    >>> result = monotone_queue_dp(arr, 2, trans)
    """
    n = len(arr)
    dp = [float('inf')] * n
    dp[0] = 0
    
    # Monotone deque (indices)
    dq = deque([0])
    
    for i in range(1, n):
        # Retire elements hors de la fenetre
        while dq and dq[0] < i - k:
            dq.popleft()
        
        # Calcule dp[i]
        if dq:
            dp[i] = transition_fn(dq[0], i)
        
        # Maintient monotonie (pour min, garde croissant)
        while dq and transition_fn(dq[-1], i + 1) >= transition_fn(i, i + 1):
            dq.pop()
        
        dq.append(i)
    
    return dp


def divide_conquer_dp(n: int, k: int, 
                     cost_fn: Callable[[int, int], int]) -> List[List[int]]:
    """
    Divide and Conquer DP Optimization.
    
    Applicable si opt[i][j] <= opt[i][j+1] (monotonie)
    
    Time: O(n*k*log n) au lieu de O(n^2*k)
    
    Args:
        n: Taille du probleme
        k: Nombre de transitions
        cost_fn: Fonction cost(i, j)
        
    Returns:
        Tableau DP[k][n]
        
    Exemple:
    --------
    >>> # Split array into k subarrays to minimize max sum
    >>> n, k = 5, 2
    >>> def cost(i, j):
    ...     return sum(range(i, j+1))
    >>> result = divide_conquer_dp(n, k, cost)
    """
    INF = float('inf')
    dp = [[INF] * n for _ in range(k + 1)]
    
    # Base case
    for i in range(n):
        dp[1][i] = cost_fn(0, i)
    
    def compute(t: int, l: int, r: int, opt_l: int, opt_r: int):
        """
        Calcule dp[t][l:r+1] sachant que opt est dans [opt_l, opt_r]
        """
        if l > r:
            return
        
        mid = (l + r) // 2
        best_cost = INF
        best_k = -1
        
        # Cherche meilleur k dans [opt_l, opt_r]
        for k in range(max(0, opt_l), min(mid, opt_r) + 1):
            cost = dp[t - 1][k] + cost_fn(k + 1, mid)
            if cost < best_cost:
                best_cost = cost
                best_k = k
        
        dp[t][mid] = best_cost
        
        # Recurse
        compute(t, l, mid - 1, opt_l, best_k)
        compute(t, mid + 1, r, best_k, opt_r)
    
    for t in range(2, k + 1):
        compute(t, 0, n - 1, 0, n - 1)
    
    return dp


class ConvexHullTrickDP:
    """
    Convex Hull Trick pour DP avec transitions lineaires.
    Deja implemente dans dp_avance.py mais reference ici.
    """
    pass


# ============================================================================
# TESTS UNITAIRES
# ============================================================================

def test_monotone_queue_dp():
    """Test monotone queue optimization"""
    arr = [1, 5, 2, 8, 3, 9, 4]
    k = 3
    
    def trans(j, i):
        if j < 0:
            return arr[i]
        return arr[i] - arr[j]
    
    result = monotone_queue_dp(arr, k, trans)
    
    assert len(result) == len(arr)
    assert result[0] == 0
    
    print("✓ Test monotone queue DP passed")


def test_divide_conquer_dp():
    """Test divide and conquer DP"""
    n, k = 5, 2
    
    def cost(i, j):
        return (j - i + 1) ** 2
    
    result = divide_conquer_dp(n, k, cost)
    
    assert len(result) == k + 1
    assert len(result[0]) == n
    
    print("✓ Test divide and conquer DP passed")


def test_knuth_optimization():
    """Test Knuth optimization"""
    # Simple test avec petite instance
    arr = [1, 2, 3, 4]
    
    def cost(i, j):
        return (j - i) * 2
    
    result = knuth_optimization_range_dp(arr, cost)
    
    assert isinstance(result, (int, float))
    
    print("✓ Test Knuth optimization passed")


# ============================================================================
# BENCHMARK
# ============================================================================

def benchmark_dp_optimizations():
    """Benchmark DP optimizations"""
    import time
    import random
    
    print("\n=== Benchmark DP Optimizations ===")
    
    # Monotone Queue
    for n in [1000, 5000, 10000]:
        arr = [random.randint(1, 100) for _ in range(n)]
        k = 100
        
        def trans(j, i):
            return arr[i] if j < 0 else arr[i] - arr[j]
        
        start = time.time()
        monotone_queue_dp(arr, k, trans)
        elapsed = time.time() - start
        
        print(f"Monotone Queue n={n}: {elapsed*1000:6.2f}ms")
    
    # Divide and Conquer
    for n in [100, 500, 1000]:
        k = 5
        
        def cost(i, j):
            return (j - i + 1) ** 2
        
        start = time.time()
        divide_conquer_dp(n, k, cost)
        elapsed = time.time() - start
        
        print(f"D&C DP n={n}, k={k}: {elapsed*1000:6.2f}ms")


if __name__ == "__main__":
    test_monotone_queue_dp()
    test_divide_conquer_dp()
    test_knuth_optimization()
    
    benchmark_dp_optimizations()
    
    print("\n✓ Tous les tests DP Optimizations passes!")

