"""
================================================================================
POLLARD'S RHO ALGORITHM
================================================================================

Description:
-----------
Algorithme de factorisation probabiliste pour trouver rapidement les facteurs
premiers de grands nombres (jusqu'a 10^18). Utilise Miller-Rabin pour tester
la primalite et Pollard Rho pour trouver les facteurs.

Complexite:
-----------
- Temps: O(n^(1/4)) attendu pour trouver un facteur
- Espace: O(log n)

Cas d'usage typiques:
--------------------
1. Factorisation de tres grands nombres
2. Problemes cryptographiques
3. Trouver diviseurs d'un nombre
4. Euler's totient pour grands nombres

Problemes classiques:
--------------------
- Project Euler (plusieurs problemes)
- Codeforces 1033D - Divisors
- SPOJ FACT0 - Integer Factorization
- AtCoder ABC 212E - Safety Journey

Auteur: Assistant CP
Date: 2025
================================================================================
"""

import random
from typing import List, Dict
from collections import Counter


def gcd(a: int, b: int) -> int:
    """Calcule le PGCD avec l'algorithme d'Euclide"""
    while b:
        a, b = b, a % b
    return a


def pow_mod(base: int, exp: int, mod: int) -> int:
    """
    Exponentiation modulaire rapide.
    
    Time: O(log exp)
    """
    result = 1
    base %= mod
    
    while exp > 0:
        if exp & 1:
            result = (result * base) % mod
        base = (base * base) % mod
        exp >>= 1
    
    return result


def miller_rabin(n: int, iterations: int = 20) -> bool:
    """
    Test de primalite de Miller-Rabin.
    
    Args:
        n: Nombre a tester
        iterations: Nombre d'iterations (plus = plus precis)
    
    Returns:
        True si probablement premier, False si compose
    
    Time: O(k * log^3 n) ou k = iterations
    
    Exemple:
    --------
    >>> miller_rabin(17)
    True
    >>> miller_rabin(18)
    False
    >>> miller_rabin(1000000007)
    True
    """
    if n < 2:
        return False
    if n == 2 or n == 3:
        return True
    if n % 2 == 0:
        return False
    
    # Ecrit n-1 comme 2^r * d
    r, d = 0, n - 1
    while d % 2 == 0:
        r += 1
        d //= 2
    
    # Witnesses a tester
    witnesses = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    for _ in range(iterations):
        a = random.choice(witnesses) if n > 40 else random.randint(2, n - 2)
        
        x = pow_mod(a, d, n)
        
        if x == 1 or x == n - 1:
            continue
        
        for _ in range(r - 1):
            x = pow_mod(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    
    return True


def pollard_rho(n: int) -> int:
    """
    Trouve un facteur non-trivial de n avec Pollard's Rho.
    
    Args:
        n: Nombre a factoriser (n > 1, n compose)
    
    Returns:
        Un facteur de n (peut etre n si echec, reessayer)
    
    Time: O(n^(1/4)) attendu
    
    Exemple:
    --------
    >>> n = 8051  # 83 * 97
    >>> factor = pollard_rho(n)
    >>> assert n % factor == 0 and factor != 1 and factor != n
    """
    if n % 2 == 0:
        return 2
    
    # Fonction f(x) = (x^2 + c) mod n
    c = random.randint(1, n - 1)
    
    x = random.randint(2, n - 1)
    y = x
    d = 1
    
    # Algorithme de Floyd (cycle detection)
    while d == 1:
        x = (x * x + c) % n
        y = (y * y + c) % n
        y = (y * y + c) % n
        
        d = gcd(abs(x - y), n)
        
        # Si on trouve n, on a echoue (changer c)
        if d == n:
            return pollard_rho(n)  # Reessayer avec nouveau c
    
    return d


def prime_factorization(n: int) -> List[int]:
    """
    Factorise completement n en facteurs premiers.
    
    Args:
        n: Nombre a factoriser
    
    Returns:
        Liste des facteurs premiers (avec repetitions)
    
    Time: O(n^(1/4) * log n) attendu
    
    Exemple:
    --------
    >>> prime_factorization(60)
    [2, 2, 3, 5]
    >>> prime_factorization(1000000007)
    [1000000007]
    """
    if n <= 1:
        return []
    
    if n == 2:
        return [2]
    
    if miller_rabin(n):
        return [n]
    
    # Trouve un facteur
    factor = pollard_rho(n)
    
    # Factorise recursivement
    return prime_factorization(factor) + prime_factorization(n // factor)


def factorize(n: int) -> Dict[int, int]:
    """
    Factorise n et retourne un dictionnaire {facteur: exposant}.
    
    Args:
        n: Nombre a factoriser
    
    Returns:
        Dictionnaire des facteurs premiers avec leurs exposants
    
    Exemple:
    --------
    >>> factorize(60)
    {2: 2, 3: 1, 5: 1}
    >>> factorize(1024)
    {2: 10}
    """
    factors = prime_factorization(n)
    return dict(Counter(factors))


def get_divisors(n: int) -> List[int]:
    """
    Trouve tous les diviseurs de n.
    
    Args:
        n: Nombre dont on veut les diviseurs
    
    Returns:
        Liste triee des diviseurs
    
    Time: O(2^k) ou k = nombre de facteurs premiers distincts
    
    Exemple:
    --------
    >>> get_divisors(12)
    [1, 2, 3, 4, 6, 12]
    >>> get_divisors(100)
    [1, 2, 4, 5, 10, 20, 25, 50, 100]
    """
    factors = factorize(n)
    
    divisors = [1]
    for prime, count in factors.items():
        new_divisors = []
        power = 1
        for _ in range(count):
            power *= prime
            for d in divisors:
                new_divisors.append(d * power)
        divisors.extend(new_divisors)
    
    return sorted(divisors)


def count_divisors(n: int) -> int:
    """
    Compte le nombre de diviseurs de n.
    
    Time: O(n^(1/4) * log n)
    
    Exemple:
    --------
    >>> count_divisors(12)
    6
    >>> count_divisors(100)
    9
    """
    factors = factorize(n)
    
    count = 1
    for exp in factors.values():
        count *= (exp + 1)
    
    return count


def sum_of_divisors(n: int) -> int:
    """
    Calcule la somme de tous les diviseurs de n.
    
    Formule: Si n = p1^a1 * p2^a2 * ... * pk^ak
             sigma(n) = (p1^(a1+1) - 1)/(p1 - 1) * ... * (pk^(ak+1) - 1)/(pk - 1)
    
    Time: O(n^(1/4) * log n)
    
    Exemple:
    --------
    >>> sum_of_divisors(12)
    28
    >>> sum_of_divisors(6)
    12
    """
    factors = factorize(n)
    
    result = 1
    for prime, exp in factors.items():
        # (p^(e+1) - 1) / (p - 1)
        result *= (pow(prime, exp + 1) - 1) // (prime - 1)
    
    return result


def euler_phi_fast(n: int) -> int:
    """
    Calcule l'indicatrice d'Euler phi(n) avec factorisation.
    
    phi(n) = n * produit((p-1)/p) pour chaque facteur premier p
    
    Time: O(n^(1/4) * log n)
    
    Exemple:
    --------
    >>> euler_phi_fast(9)
    6
    >>> euler_phi_fast(10)
    4
    """
    factors = factorize(n)
    
    result = n
    for prime in factors:
        result = result * (prime - 1) // prime
    
    return result


# ============================================================================
# TESTS UNITAIRES
# ============================================================================

def test_miller_rabin():
    """Test Miller-Rabin pour primalite"""
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 97, 1000000007]
    composites = [4, 6, 8, 9, 10, 12, 15, 100, 1000]
    
    for p in primes:
        assert miller_rabin(p), f"{p} devrait etre premier"
    
    for c in composites:
        assert not miller_rabin(c), f"{c} ne devrait pas etre premier"
    
    print("✓ Test Miller-Rabin passed")


def test_pollard_rho():
    """Test Pollard Rho pour factorisation"""
    # Nombres composes simples
    test_cases = [
        (15, {3, 5}),
        (21, {3, 7}),
        (35, {5, 7}),
        (143, {11, 13}),
    ]
    
    for n, expected_factors in test_cases:
        factor = pollard_rho(n)
        assert factor in expected_factors or n // factor in expected_factors
    
    print("✓ Test Pollard Rho passed")


def test_prime_factorization():
    """Test factorisation complete"""
    test_cases = [
        (1, []),
        (2, [2]),
        (12, [2, 2, 3]),
        (60, [2, 2, 3, 5]),
        (100, [2, 2, 5, 5]),
        (1024, [2] * 10),
    ]
    
    for n, expected in test_cases:
        result = sorted(prime_factorization(n))
        assert result == expected, f"factorize({n}) = {result}, expected {expected}"
    
    print("✓ Test prime factorization passed")


def test_factorize():
    """Test factorisation avec exposants"""
    test_cases = [
        (12, {2: 2, 3: 1}),
        (60, {2: 2, 3: 1, 5: 1}),
        (1024, {2: 10}),
    ]
    
    for n, expected in test_cases:
        result = factorize(n)
        assert result == expected
    
    print("✓ Test factorize passed")


def test_get_divisors():
    """Test enumeration des diviseurs"""
    test_cases = [
        (1, [1]),
        (12, [1, 2, 3, 4, 6, 12]),
        (100, [1, 2, 4, 5, 10, 20, 25, 50, 100]),
    ]
    
    for n, expected in test_cases:
        result = get_divisors(n)
        assert result == expected
    
    print("✓ Test get divisors passed")


def test_count_divisors():
    """Test comptage des diviseurs"""
    assert count_divisors(1) == 1
    assert count_divisors(12) == 6
    assert count_divisors(100) == 9
    
    print("✓ Test count divisors passed")


def test_sum_of_divisors():
    """Test somme des diviseurs"""
    assert sum_of_divisors(6) == 12  # 1+2+3+6
    assert sum_of_divisors(12) == 28  # 1+2+3+4+6+12
    
    print("✓ Test sum of divisors passed")


def test_euler_phi_fast():
    """Test phi d'Euler rapide"""
    assert euler_phi_fast(1) == 1
    assert euler_phi_fast(9) == 6
    assert euler_phi_fast(10) == 4
    
    print("✓ Test Euler phi fast passed")


def test_large_numbers():
    """Test avec grands nombres"""
    # Nombre compose de deux grands premiers
    n = 10007 * 10009  # 100160063
    
    factors = sorted(prime_factorization(n))
    assert len(factors) == 2
    assert factors[0] * factors[1] == n
    
    print("✓ Test large numbers passed")


# ============================================================================
# BENCHMARK
# ============================================================================

def benchmark_pollard_rho():
    """Benchmark Pollard Rho"""
    import time
    
    print("\n=== Benchmark Pollard Rho ===")
    
    test_numbers = [
        1000003,  # Premier
        1000000007,  # Premier
        10007 * 10009,  # Produit de 2 premiers
        1000000000000037,  # Grand premier
        999999999999999989,  # Tres grand premier
    ]
    
    for n in test_numbers:
        start = time.time()
        
        if miller_rabin(n):
            result = [n]
        else:
            result = prime_factorization(n)
        
        elapsed = time.time() - start
        
        print(f"n={n:20d}: factors={result}, time={elapsed*1000:8.3f}ms")


def benchmark_divisors():
    """Benchmark enumeration des diviseurs"""
    import time
    
    print("\n=== Benchmark Divisors ===")
    
    test_numbers = [100, 1000, 10000, 100000]
    
    for n in test_numbers:
        start = time.time()
        divisors = get_divisors(n)
        elapsed = time.time() - start
        
        print(f"n={n:6d}: {len(divisors):4d} divisors, time={elapsed*1000:6.2f}ms")


if __name__ == "__main__":
    # Tests
    test_miller_rabin()
    test_pollard_rho()
    test_prime_factorization()
    test_factorize()
    test_get_divisors()
    test_count_divisors()
    test_sum_of_divisors()
    test_euler_phi_fast()
    test_large_numbers()
    
    # Benchmarks
    benchmark_pollard_rho()
    benchmark_divisors()
    
    print("\n✓ Tous les tests Pollard Rho passes!")

