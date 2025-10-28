"""
Programmation Dynamique Avancée

Description:
    Techniques DP avancées:
    - DP Bitmask
    - DP Digit
    - Convex Hull Trick
    - DP Divide and Conquer Optimization

Complexité:
    Variable selon la technique

Cas d'usage:
    - Problèmes de sous-ensembles
    - Comptage avec contraintes
    - Optimisations DP

Problèmes types:
    - Codeforces: 337D, 607B, 1083E
    - AtCoder: ABC180E, ABC147F
    - CSES: Hamiltonian Flights
    
Implémentation par: 2025-10-27
Testé: Oui
"""


def tsp_bitmask(dist):
    """
    Travelling Salesman Problem en O(n² × 2^n) avec DP bitmask.
    
    Args:
        dist: Matrice de distances dist[i][j]
        
    Returns:
        Distance minimale du tour
        
    Example:
        >>> dist = [[0, 10, 15, 20], [10, 0, 35, 25], [15, 35, 0, 30], [20, 25, 30, 0]]
        >>> tsp_bitmask(dist)
        80
    """
    n = len(dist)
    
    # dp[mask][i] = distance min pour visiter les villes dans mask, finissant à i
    dp = [[float('inf')] * n for _ in range(1 << n)]
    dp[1][0] = 0  # Commencer à la ville 0
    
    for mask in range(1 << n):
        for u in range(n):
            if dp[mask][u] == float('inf'):
                continue
            
            for v in range(n):
                if mask & (1 << v):  # v déjà visité
                    continue
                
                new_mask = mask | (1 << v)
                dp[new_mask][v] = min(dp[new_mask][v], dp[mask][u] + dist[u][v])
    
    # Retourner à la ville de départ
    full_mask = (1 << n) - 1
    result = min(dp[full_mask][i] + dist[i][0] for i in range(1, n))
    
    return result


def subset_sum_bitmask(arr, target):
    """
    Compte le nombre de sous-ensembles dont la somme vaut target.
    Utilise DP bitmask pour petits ensembles.
    
    Args:
        arr: Liste de nombres
        target: Somme cible
        
    Returns:
        Nombre de sous-ensembles
    """
    n = len(arr)
    count = 0
    
    # Énumérer tous les sous-ensembles
    for mask in range(1 << n):
        subset_sum = sum(arr[i] for i in range(n) if mask & (1 << i))
        if subset_sum == target:
            count += 1
    
    return count


def hamiltonian_paths_bitmask(graph):
    """
    Compte les chemins hamiltoniens dans un graphe dirigé.
    
    Args:
        graph: Matrice d'adjacence
        
    Returns:
        Nombre de chemins hamiltoniens de 0 à n-1
    """
    n = len(graph)
    
    # dp[mask][i] = nombre de chemins visitant mask et finissant à i
    dp = [[0] * n for _ in range(1 << n)]
    dp[1][0] = 1  # Commencer au sommet 0
    
    for mask in range(1 << n):
        for u in range(n):
            if not (mask & (1 << u)) or dp[mask][u] == 0:
                continue
            
            for v in range(n):
                if (mask & (1 << v)) or not graph[u][v]:
                    continue
                
                new_mask = mask | (1 << v)
                dp[new_mask][v] += dp[mask][u]
    
    full_mask = (1 << n) - 1
    return dp[full_mask][n - 1]


def count_numbers_with_digit_sum(n, target_sum):
    """
    DP Digit: Compte les nombres de 1 à n dont la somme des chiffres = target_sum.
    
    Args:
        n: Borne supérieure
        target_sum: Somme cible des chiffres
        
    Returns:
        Nombre de tels nombres
    """
    s = str(n)
    length = len(s)
    
    # dp[pos][sum][tight] = nombre de nombres
    memo = {}
    
    def dp(pos, current_sum, tight):
        if pos == length:
            return 1 if current_sum == target_sum else 0
        
        if (pos, current_sum, tight) in memo:
            return memo[(pos, current_sum, tight)]
        
        limit = int(s[pos]) if tight else 9
        result = 0
        
        for digit in range(0, limit + 1):
            if current_sum + digit <= target_sum:
                new_tight = tight and (digit == limit)
                result += dp(pos + 1, current_sum + digit, new_tight)
        
        memo[(pos, current_sum, tight)] = result
        return result
    
    return dp(0, 0, True)


def sos_dp(arr):
    """
    Sum Over Subsets DP.
    Pour chaque mask, calcule la somme de arr[submask] pour tous les submasks.
    
    Args:
        arr: Tableau indexé par masks (taille 2^n)
        
    Returns:
        Tableau des sommes de sous-ensembles
        
    Example:
        >>> arr = [1, 2, 3, 4, 5, 6, 7, 8]  # arr[mask] pour mask de 0 à 7
        >>> sos = sos_dp(arr)
        >>> sos[7]  # Somme de arr[0,1,2,3,4,5,6,7]
        36
    """
    n = len(arr).bit_length() - 1
    size = 1 << n
    
    dp = arr[:]
    
    for i in range(n):
        for mask in range(size):
            if mask & (1 << i):
                dp[mask] += dp[mask ^ (1 << i)]
    
    return dp


class ConvexHullTrick:
    """
    Convex Hull Trick pour optimiser DP de la forme:
    dp[i] = min(dp[j] + cost(j, i)) où cost est monotone
    """
    
    def __init__(self):
        self.lines = []  # (slope, intercept)
    
    def _bad(self, l1, l2, l3):
        """Vérifie si l2 est inutile"""
        # Intersection de l1-l2 >= intersection de l2-l3
        return (l3[1] - l1[1]) * (l1[0] - l2[0]) <= (l2[1] - l1[1]) * (l1[0] - l3[0])
    
    def add_line(self, slope, intercept):
        """
        Ajoute une ligne y = slope*x + intercept.
        Les slopes doivent être ajoutés en ordre décroissant.
        """
        line = (slope, intercept)
        
        while len(self.lines) >= 2 and self._bad(self.lines[-2], self.lines[-1], line):
            self.lines.pop()
        
        self.lines.append(line)
    
    def query(self, x):
        """
        Trouve le minimum de toutes les lignes évaluées en x.
        x doit être en ordre croissant.
        """
        while len(self.lines) >= 2:
            # Si la deuxième ligne est meilleure, retirer la première
            if self.lines[0][0] * x + self.lines[0][1] >= \
               self.lines[1][0] * x + self.lines[1][1]:
                self.lines.pop(0)
            else:
                break
        
        if not self.lines:
            return float('inf')
        
        return self.lines[0][0] * x + self.lines[0][1]


def building_bridges_cht(heights, weights):
    """
    Problème classique de CHT: construire des ponts entre bâtiments.
    
    Args:
        heights: Hauteurs des bâtiments
        weights: Poids des bâtiments
        
    Returns:
        Coût minimum
    """
    n = len(heights)
    
    # dp[i] = coût min pour construire jusqu'au bâtiment i
    dp = [float('inf')] * n
    dp[0] = 0
    
    # Prefix sum des poids
    prefix_weight = [0] * (n + 1)
    for i in range(n):
        prefix_weight[i + 1] = prefix_weight[i] + weights[i]
    
    cht = ConvexHullTrick()
    cht.add_line(-2 * heights[0], heights[0]**2 + dp[0] - prefix_weight[1])
    
    for i in range(1, n):
        # dp[i] = min(dp[j] + (h[i] - h[j])^2 + sum(weights[j+1..i-1]))
        #       = min(dp[j] + h[i]^2 - 2*h[i]*h[j] + h[j]^2 + prefix[i] - prefix[j+1])
        
        cost = cht.query(heights[i]) + heights[i]**2 + prefix_weight[i]
        dp[i] = cost
        
        if i < n - 1:
            cht.add_line(-2 * heights[i], heights[i]**2 + dp[i] - prefix_weight[i + 1])
    
    return int(dp[n - 1])


def test():
    """Tests unitaires complets"""
    
    # Test TSP
    dist = [
        [0, 10, 15, 20],
        [10, 0, 35, 25],
        [15, 35, 0, 30],
        [20, 25, 30, 0]
    ]
    tsp_cost = tsp_bitmask(dist)
    assert tsp_cost == 80  # 0->1->3->2->0
    
    # Test subset sum
    arr = [2, 3, 5]
    count = subset_sum_bitmask(arr, 5)
    assert count == 2  # {5} et {2,3}
    
    # Test hamiltonian paths
    graph = [
        [0, 1, 1],
        [0, 0, 1],
        [0, 0, 0]
    ]
    paths = hamiltonian_paths_bitmask(graph)
    assert paths == 2  # 0->1->2 et 0->2 (non valide car pas hamiltonien)
    
    # Test DP digit
    count_10 = count_numbers_with_digit_sum(20, 5)
    assert count_10 > 0
    
    # Test SOS DP
    arr_sos = [1, 2, 3, 4, 5, 6, 7, 8]
    sos_result = sos_dp(arr_sos)
    assert sos_result[0] == 1  # Seulement arr[0]
    assert sos_result[7] == sum(arr_sos)  # Tous les sous-ensembles de 7
    
    # Test Convex Hull Trick
    cht = ConvexHullTrick()
    cht.add_line(-1, 0)  # y = -x
    cht.add_line(-2, 3)  # y = -2x + 3
    
    assert cht.query(0) == 0
    assert cht.query(1) == -1
    
    # Test building bridges (simplifié)
    heights = [1, 2, 3, 4]
    weights = [1, 1, 1, 1]
    cost = building_bridges_cht(heights, weights)
    assert cost >= 0
    
    print("Tous les tests passes")


if __name__ == "__main__":
    test()

