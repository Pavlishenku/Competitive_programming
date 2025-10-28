"""
Binary Search et variantes

Description:
    Recherche binaire et ses variantes pour différents cas d'usage:
    - Binary search classique
    - Lower bound / Upper bound
    - Binary search sur la réponse
    - Ternary search

Complexité:
    - Binary search: O(log n)
    - Ternary search: O(log n)

Cas d'usage:
    - Recherche dans tableau trié
    - Optimisation sur intervalle
    - Recherche de seuil
    - Minimisation/maximisation unimodale
    
Problèmes types:
    - Codeforces: 279B, 165B, 460C
    - AtCoder: ABC077C, ABC146D
    - CSES: Factory Machines
    
Implémentation par: 2025-10-27
Testé: Oui
"""


def binary_search(arr, target):
    """
    Binary search classique.
    
    Args:
        arr: Tableau trié
        target: Valeur à chercher
        
    Returns:
        Index de target, ou -1 si non trouvé
        
    Example:
        >>> binary_search([1, 3, 5, 7, 9], 5)
        2
    """
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = (left + right) // 2
        
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1


def lower_bound(arr, target):
    """
    Trouve le premier élément >= target (lower_bound C++).
    
    Args:
        arr: Tableau trié
        target: Valeur cible
        
    Returns:
        Index du premier élément >= target, ou len(arr)
        
    Example:
        >>> lower_bound([1, 2, 2, 3, 5], 2)
        1
    """
    left, right = 0, len(arr)
    
    while left < right:
        mid = (left + right) // 2
        
        if arr[mid] < target:
            left = mid + 1
        else:
            right = mid
    
    return left


def upper_bound(arr, target):
    """
    Trouve le premier élément > target (upper_bound C++).
    
    Args:
        arr: Tableau trié
        target: Valeur cible
        
    Returns:
        Index du premier élément > target, ou len(arr)
        
    Example:
        >>> upper_bound([1, 2, 2, 3, 5], 2)
        3
    """
    left, right = 0, len(arr)
    
    while left < right:
        mid = (left + right) // 2
        
        if arr[mid] <= target:
            left = mid + 1
        else:
            right = mid
    
    return left


def binary_search_first(arr, target):
    """
    Trouve la première occurrence de target.
    
    Args:
        arr: Tableau trié
        target: Valeur à chercher
        
    Returns:
        Index de la première occurrence, ou -1
    """
    idx = lower_bound(arr, target)
    
    if idx < len(arr) and arr[idx] == target:
        return idx
    return -1


def binary_search_last(arr, target):
    """
    Trouve la dernière occurrence de target.
    
    Args:
        arr: Tableau trié
        target: Valeur à chercher
        
    Returns:
        Index de la dernière occurrence, ou -1
    """
    idx = upper_bound(arr, target)
    
    if idx > 0 and arr[idx - 1] == target:
        return idx - 1
    return -1


def binary_search_answer(predicate, left, right):
    """
    Binary search sur la réponse.
    Trouve le plus grand x tel que predicate(x) est True.
    
    Args:
        predicate: Fonction booléenne monotone
        left: Borne inférieure
        right: Borne supérieure
        
    Returns:
        Plus grand x avec predicate(x) = True, ou left-1 si aucun
        
    Example:
        >>> def can_do(x): return x * x <= 100
        >>> binary_search_answer(can_do, 0, 100)
        10
    """
    result = left - 1
    
    while left <= right:
        mid = (left + right) // 2
        
        if predicate(mid):
            result = mid
            left = mid + 1
        else:
            right = mid - 1
    
    return result


def binary_search_answer_min(predicate, left, right):
    """
    Binary search sur la réponse (minimisation).
    Trouve le plus petit x tel que predicate(x) est True.
    
    Args:
        predicate: Fonction booléenne monotone
        left: Borne inférieure
        right: Borne supérieure
        
    Returns:
        Plus petit x avec predicate(x) = True, ou right+1 si aucun
    """
    result = right + 1
    
    while left <= right:
        mid = (left + right) // 2
        
        if predicate(mid):
            result = mid
            right = mid - 1
        else:
            left = mid + 1
    
    return result


def binary_search_float(predicate, left, right, epsilon=1e-9):
    """
    Binary search sur les réels.
    
    Args:
        predicate: Fonction booléenne
        left: Borne inférieure
        right: Borne supérieure
        epsilon: Précision
        
    Returns:
        Valeur réelle optimale
        
    Example:
        >>> def f(x): return x * x <= 2
        >>> result = binary_search_float(f, 0, 2)
        >>> abs(result - 1.414) < 0.01
        True
    """
    while right - left > epsilon:
        mid = (left + right) / 2
        
        if predicate(mid):
            left = mid
        else:
            right = mid
    
    return (left + right) / 2


def ternary_search(f, left, right, is_minimum=True, epsilon=1e-9):
    """
    Ternary search pour trouver min/max d'une fonction unimodale.
    
    Args:
        f: Fonction unimodale
        left: Borne gauche
        right: Borne droite
        is_minimum: True pour minimum, False pour maximum
        epsilon: Précision
        
    Returns:
        x optimal
        
    Example:
        >>> def parabola(x): return (x - 5) ** 2
        >>> result = ternary_search(parabola, 0, 10)
        >>> abs(result - 5) < 0.001
        True
    """
    while right - left > epsilon:
        mid1 = left + (right - left) / 3
        mid2 = right - (right - left) / 3
        
        f1 = f(mid1)
        f2 = f(mid2)
        
        if is_minimum:
            if f1 < f2:
                right = mid2
            else:
                left = mid1
        else:
            if f1 > f2:
                right = mid2
            else:
                left = mid1
    
    return (left + right) / 2


def ternary_search_discrete(f, left, right, is_minimum=True):
    """
    Ternary search sur entiers.
    
    Args:
        f: Fonction unimodale sur entiers
        left: Borne gauche
        right: Borne droite
        is_minimum: True pour minimum, False pour maximum
        
    Returns:
        x optimal (entier)
    """
    while right - left > 2:
        mid1 = left + (right - left) // 3
        mid2 = right - (right - left) // 3
        
        f1 = f(mid1)
        f2 = f(mid2)
        
        if is_minimum:
            if f1 < f2:
                right = mid2
            else:
                left = mid1
        else:
            if f1 > f2:
                right = mid2
            else:
                left = mid1
    
    # Vérifier les candidats restants
    best_x = left
    best_val = f(left)
    
    for x in range(left + 1, right + 1):
        val = f(x)
        if (is_minimum and val < best_val) or (not is_minimum and val > best_val):
            best_val = val
            best_x = x
    
    return best_x


def exponential_search(arr, target):
    """
    Exponential search (pour tableaux non bornés ou très grands).
    
    Args:
        arr: Tableau trié
        target: Valeur à chercher
        
    Returns:
        Index de target, ou -1
    """
    if not arr:
        return -1
    
    if arr[0] == target:
        return 0
    
    # Trouver la plage
    i = 1
    while i < len(arr) and arr[i] <= target:
        i *= 2
    
    # Binary search dans [i//2, min(i, len(arr)-1)]
    left = i // 2
    right = min(i, len(arr) - 1)
    
    while left <= right:
        mid = (left + right) // 2
        
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1


def test():
    """Tests unitaires complets"""
    
    # Test binary search
    arr = [1, 3, 5, 7, 9, 11]
    assert binary_search(arr, 5) == 2
    assert binary_search(arr, 6) == -1
    assert binary_search(arr, 1) == 0
    assert binary_search(arr, 11) == 5
    
    # Test lower_bound
    arr2 = [1, 2, 2, 3, 5]
    assert lower_bound(arr2, 2) == 1
    assert lower_bound(arr2, 4) == 4
    assert lower_bound(arr2, 0) == 0
    assert lower_bound(arr2, 6) == 5
    
    # Test upper_bound
    assert upper_bound(arr2, 2) == 3
    assert upper_bound(arr2, 3) == 4
    assert upper_bound(arr2, 5) == 5
    
    # Test first/last occurrence
    assert binary_search_first([1, 2, 2, 2, 3], 2) == 1
    assert binary_search_last([1, 2, 2, 2, 3], 2) == 3
    
    # Test binary search sur la réponse
    def can_do(x):
        return x * x <= 100
    
    result = binary_search_answer(can_do, 0, 100)
    assert result == 10
    
    # Test binary search float
    def f_float(x):
        return x * x <= 2
    
    result_float = binary_search_float(f_float, 0, 2)
    assert abs(result_float - 1.414213) < 0.001
    
    # Test ternary search
    def parabola(x):
        return (x - 5) ** 2
    
    minimum = ternary_search(parabola, 0, 10)
    assert abs(minimum - 5) < 0.001
    
    # Test ternary search discrete
    def discrete_f(x):
        return abs(x - 7)
    
    min_x = ternary_search_discrete(discrete_f, 0, 20)
    assert min_x == 7
    
    # Test exponential search
    arr3 = list(range(1, 1001, 2))  # [1, 3, 5, ..., 999]
    assert exponential_search(arr3, 501) == 250
    assert exponential_search(arr3, 1) == 0
    assert exponential_search(arr3, 500) == -1
    
    # Test edge cases
    assert binary_search([], 5) == -1
    assert binary_search([5], 5) == 0
    assert lower_bound([1, 2, 3], 0) == 0
    assert upper_bound([1, 2, 3], 3) == 3
    
    print("Tous les tests passes")


if __name__ == "__main__":
    test()

