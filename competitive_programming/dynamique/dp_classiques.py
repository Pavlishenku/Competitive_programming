"""
Programmation Dynamique Classique

Description:
    Algorithmes DP fondamentaux:
    - Knapsack (0/1 et unbounded)
    - Longest Increasing Subsequence (LIS)
    - Longest Common Subsequence (LCS)
    - Edit Distance
    - Coin Change

Complexité:
    Variable selon l'algorithme (généralement O(n²) ou O(nW))

Cas d'usage:
    - Problèmes d'optimisation
    - Séquences et sous-séquences
    - Partitionnement
    - Chemins dans grilles
    
Problèmes types:
    - Codeforces: 189A, 231C, 455A
    - AtCoder: ABC129C, ABC143D
    - CSES: DP section
    
Implémentation par: 2025-10-27
Testé: Oui
"""

from bisect import bisect_left


def knapsack_01(weights, values, capacity):
    """
    Knapsack 0/1: chaque objet peut être pris au plus une fois.
    
    Args:
        weights: Liste des poids
        values: Liste des valeurs
        capacity: Capacité du sac
        
    Returns:
        Valeur maximale
        
    Example:
        >>> knapsack_01([2, 3, 4, 5], [3, 4, 5, 6], 5)
        7
    """
    n = len(weights)
    dp = [0] * (capacity + 1)
    
    for i in range(n):
        # Parcourir de droite à gauche pour éviter de prendre deux fois
        for w in range(capacity, weights[i] - 1, -1):
            dp[w] = max(dp[w], dp[w - weights[i]] + values[i])
    
    return dp[capacity]


def knapsack_01_with_items(weights, values, capacity):
    """
    Knapsack 0/1 qui retourne aussi les objets sélectionnés.
    
    Args:
        weights: Liste des poids
        values: Liste des valeurs
        capacity: Capacité
        
    Returns:
        Tuple (valeur_max, liste_indices_objets)
    """
    n = len(weights)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    
    for i in range(1, n + 1):
        for w in range(capacity + 1):
            # Ne pas prendre l'objet i-1
            dp[i][w] = dp[i-1][w]
            
            # Prendre l'objet i-1 si possible
            if w >= weights[i-1]:
                dp[i][w] = max(dp[i][w], 
                              dp[i-1][w - weights[i-1]] + values[i-1])
    
    # Reconstruire la solution
    w = capacity
    items = []
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i-1][w]:
            items.append(i-1)
            w -= weights[i-1]
    
    items.reverse()
    return (dp[n][capacity], items)


def knapsack_unbounded(weights, values, capacity):
    """
    Knapsack unbounded: chaque objet peut être pris infiniment.
    
    Args:
        weights: Liste des poids
        values: Liste des valeurs
        capacity: Capacité
        
    Returns:
        Valeur maximale
    """
    dp = [0] * (capacity + 1)
    
    for w in range(1, capacity + 1):
        for i in range(len(weights)):
            if weights[i] <= w:
                dp[w] = max(dp[w], dp[w - weights[i]] + values[i])
    
    return dp[capacity]


def lis_n_squared(arr):
    """
    Longest Increasing Subsequence en O(n²).
    
    Args:
        arr: Liste de nombres
        
    Returns:
        Longueur de la plus longue sous-séquence croissante
        
    Example:
        >>> lis_n_squared([10, 9, 2, 5, 3, 7, 101, 18])
        4
    """
    n = len(arr)
    if n == 0:
        return 0
    
    dp = [1] * n
    
    for i in range(1, n):
        for j in range(i):
            if arr[j] < arr[i]:
                dp[i] = max(dp[i], dp[j] + 1)
    
    return max(dp)


def lis_nlogn(arr):
    """
    Longest Increasing Subsequence en O(n log n).
    Utilise binary search.
    
    Args:
        arr: Liste de nombres
        
    Returns:
        Longueur de la LIS
    """
    if not arr:
        return 0
    
    # tails[i] = plus petit élément terminant une LIS de longueur i+1
    tails = []
    
    for num in arr:
        pos = bisect_left(tails, num)
        if pos == len(tails):
            tails.append(num)
        else:
            tails[pos] = num
    
    return len(tails)


def lis_with_sequence(arr):
    """
    LIS qui retourne aussi la séquence.
    
    Args:
        arr: Liste de nombres
        
    Returns:
        Liste représentant la LIS
    """
    if not arr:
        return []
    
    n = len(arr)
    tails = []
    parent = [-1] * n
    indices = []
    
    for i, num in enumerate(arr):
        pos = bisect_left(tails, num)
        if pos == len(tails):
            tails.append(num)
            indices.append(i)
        else:
            tails[pos] = num
            indices[pos] = i
        
        if pos > 0:
            parent[i] = indices[pos - 1]
    
    # Reconstruire la séquence
    result = []
    k = indices[-1]
    while k != -1:
        result.append(arr[k])
        k = parent[k]
    
    result.reverse()
    return result


def lcs(s1, s2):
    """
    Longest Common Subsequence.
    
    Args:
        s1: Première chaîne
        s2: Deuxième chaîne
        
    Returns:
        Longueur de la plus longue sous-séquence commune
        
    Example:
        >>> lcs("ABCDGH", "AEDFHR")
        3
    """
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    return dp[m][n]


def lcs_string(s1, s2):
    """
    LCS qui retourne la sous-séquence.
    
    Args:
        s1: Première chaîne
        s2: Deuxième chaîne
        
    Returns:
        La LCS sous forme de chaîne
    """
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    # Reconstruire la LCS
    result = []
    i, j = m, n
    while i > 0 and j > 0:
        if s1[i-1] == s2[j-1]:
            result.append(s1[i-1])
            i -= 1
            j -= 1
        elif dp[i-1][j] > dp[i][j-1]:
            i -= 1
        else:
            j -= 1
    
    return ''.join(reversed(result))


def edit_distance(s1, s2):
    """
    Distance d'édition (Levenshtein distance).
    Nombre minimum d'opérations (insertion, suppression, substitution).
    
    Args:
        s1: Première chaîne
        s2: Deuxième chaîne
        
    Returns:
        Distance d'édition
        
    Example:
        >>> edit_distance("kitten", "sitting")
        3
    """
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Initialisation
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(
                    dp[i-1][j],      # Suppression
                    dp[i][j-1],      # Insertion
                    dp[i-1][j-1]     # Substitution
                )
    
    return dp[m][n]


def coin_change_min(coins, amount):
    """
    Nombre minimum de pièces pour faire un montant.
    
    Args:
        coins: Liste des valeurs de pièces
        amount: Montant cible
        
    Returns:
        Nombre minimum de pièces, ou -1 si impossible
        
    Example:
        >>> coin_change_min([1, 2, 5], 11)
        3
    """
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    
    for a in range(1, amount + 1):
        for coin in coins:
            if coin <= a:
                dp[a] = min(dp[a], dp[a - coin] + 1)
    
    return dp[amount] if dp[amount] != float('inf') else -1


def coin_change_ways(coins, amount):
    """
    Nombre de façons de faire un montant avec des pièces.
    
    Args:
        coins: Liste des valeurs de pièces
        amount: Montant cible
        
    Returns:
        Nombre de façons
    """
    dp = [0] * (amount + 1)
    dp[0] = 1
    
    for coin in coins:
        for a in range(coin, amount + 1):
            dp[a] += dp[a - coin]
    
    return dp[amount]


def max_subarray_sum(arr):
    """
    Maximum Subarray Sum (Kadane's algorithm).
    
    Args:
        arr: Liste de nombres
        
    Returns:
        Somme maximale d'un sous-tableau contigu
        
    Example:
        >>> max_subarray_sum([-2, 1, -3, 4, -1, 2, 1, -5, 4])
        6
    """
    if not arr:
        return 0
    
    max_ending_here = max_so_far = arr[0]
    
    for num in arr[1:]:
        max_ending_here = max(num, max_ending_here + num)
        max_so_far = max(max_so_far, max_ending_here)
    
    return max_so_far


def test():
    """Tests unitaires complets"""
    
    # Test Knapsack 0/1
    value = knapsack_01([2, 3, 4, 5], [3, 4, 5, 6], 5)
    assert value == 7  # Objets 0 et 1
    
    value2, items = knapsack_01_with_items([2, 3, 4, 5], [3, 4, 5, 6], 5)
    assert value2 == 7
    
    # Test Knapsack unbounded
    value3 = knapsack_unbounded([1, 2, 3], [10, 15, 40], 5)
    assert value3 >= 50  # 5 objets de poids 1
    
    # Test LIS
    assert lis_n_squared([10, 9, 2, 5, 3, 7, 101, 18]) == 4
    assert lis_nlogn([10, 9, 2, 5, 3, 7, 101, 18]) == 4
    
    lis_seq = lis_with_sequence([10, 9, 2, 5, 3, 7, 101, 18])
    assert len(lis_seq) == 4
    
    # Test LCS
    assert lcs("ABCDGH", "AEDFHR") == 3
    lcs_str = lcs_string("ABCDGH", "AEDFHR")
    assert len(lcs_str) == 3
    
    # Test Edit Distance
    assert edit_distance("kitten", "sitting") == 3
    assert edit_distance("abc", "abc") == 0
    
    # Test Coin Change
    assert coin_change_min([1, 2, 5], 11) == 3
    assert coin_change_min([2], 3) == -1
    
    assert coin_change_ways([1, 2, 5], 5) == 4
    
    # Test Max Subarray Sum
    assert max_subarray_sum([-2, 1, -3, 4, -1, 2, 1, -5, 4]) == 6
    assert max_subarray_sum([1, 2, 3, 4]) == 10
    assert max_subarray_sum([-1, -2, -3]) == -1
    
    # Test edge cases
    assert lis_nlogn([]) == 0
    assert lcs("", "abc") == 0
    assert edit_distance("", "") == 0
    assert coin_change_ways([1], 0) == 1
    
    print("Tous les tests passes")


if __name__ == "__main__":
    test()

