"""
Nombres Premiers (Sieve of Eratosthenes, Factorisation)

Description:
    Algorithmes pour travailler avec les nombres premiers:
    - Crible d'Ératosthène
    - Test de primalité
    - Factorisation
    - Miller-Rabin (test probabiliste)

Complexité:
    - Crible: O(n log log n)
    - Factorisation: O(√n)
    - Miller-Rabin: O(k log³ n) où k = nombre de tests

Cas d'usage:
    - Problèmes de divisibilité
    - Factorisation
    - Cryptographie
    - Théorie des nombres
    
Problèmes types:
    - Codeforces: 17A, 154B, 776B
    - AtCoder: ABC149D, ABC152E
    - CSES: Prime Multiples
    
Implémentation par: 2025-10-27
Testé: Oui
"""

import random


def sieve_of_eratosthenes(n):
    """
    Crible d'Ératosthène pour trouver tous les premiers jusqu'à n.
    
    Args:
        n: Limite supérieure
        
    Returns:
        Liste des nombres premiers <= n
        
    Example:
        >>> sieve_of_eratosthenes(20)
        [2, 3, 5, 7, 11, 13, 17, 19]
    """
    if n < 2:
        return []
    
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, n + 1, i):
                is_prime[j] = False
    
    return [i for i in range(n + 1) if is_prime[i]]


def sieve_bool(n):
    """
    Retourne un tableau booléen is_prime[i] = True si i est premier.
    Plus efficace en mémoire pour requêtes multiples.
    
    Args:
        n: Limite supérieure
        
    Returns:
        Liste booléenne où is_prime[i] indique si i est premier
    """
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, n + 1, i):
                is_prime[j] = False
    
    return is_prime


def smallest_prime_factor(n):
    """
    Calcule le plus petit facteur premier pour tous les nombres jusqu'à n.
    Utile pour factorisation rapide multiple.
    
    Args:
        n: Limite supérieure
        
    Returns:
        Liste où spf[i] = plus petit facteur premier de i
    """
    spf = list(range(n + 1))
    
    for i in range(2, int(n**0.5) + 1):
        if spf[i] == i:  # i est premier
            for j in range(i*i, n + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    return spf


def prime_factorization(n):
    """
    Factorisation en nombres premiers.
    
    Args:
        n: Nombre à factoriser
        
    Returns:
        Dict {facteur: exposant}
        
    Example:
        >>> prime_factorization(60)
        {2: 2, 3: 1, 5: 1}
    """
    factors = {}
    d = 2
    
    while d * d <= n:
        while n % d == 0:
            factors[d] = factors.get(d, 0) + 1
            n //= d
        d += 1
    
    if n > 1:
        factors[n] = factors.get(n, 0) + 1
    
    return factors


def prime_factorization_list(n):
    """
    Retourne la liste des facteurs premiers (avec répétitions).
    
    Args:
        n: Nombre à factoriser
        
    Returns:
        Liste des facteurs premiers
        
    Example:
        >>> prime_factorization_list(60)
        [2, 2, 3, 5]
    """
    factors = []
    d = 2
    
    while d * d <= n:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 1
    
    if n > 1:
        factors.append(n)
    
    return factors


def is_prime(n):
    """
    Test de primalité simple.
    
    Args:
        n: Nombre à tester
        
    Returns:
        True si n est premier, False sinon
    """
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    
    return True


def miller_rabin(n, k=5):
    """
    Test de primalité de Miller-Rabin (probabiliste).
    Plus rapide pour grands nombres.
    
    Args:
        n: Nombre à tester
        k: Nombre de tests (plus k est grand, plus c'est précis)
        
    Returns:
        True si n est probablement premier, False si composé
    """
    if n < 2:
        return False
    if n == 2 or n == 3:
        return True
    if n % 2 == 0:
        return False
    
    # Écrire n-1 comme 2^r * d
    r, d = 0, n - 1
    while d % 2 == 0:
        r += 1
        d //= 2
    
    # Test k fois
    for _ in range(k):
        a = random.randrange(2, n - 1)
        x = pow(a, d, n)
        
        if x == 1 or x == n - 1:
            continue
        
        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    
    return True


def count_divisors(n):
    """
    Compte le nombre de diviseurs de n.
    
    Args:
        n: Nombre
        
    Returns:
        Nombre de diviseurs
        
    Example:
        >>> count_divisors(12)
        6
    """
    factors = prime_factorization(n)
    count = 1
    
    for exp in factors.values():
        count *= (exp + 1)
    
    return count


def sum_of_divisors(n):
    """
    Calcule la somme des diviseurs de n.
    
    Args:
        n: Nombre
        
    Returns:
        Somme des diviseurs
    """
    factors = prime_factorization(n)
    result = 1
    
    for prime, exp in factors.items():
        # Somme géométrique: (p^(e+1) - 1) / (p - 1)
        result *= (pow(prime, exp + 1) - 1) // (prime - 1)
    
    return result


def euler_phi(n):
    """
    Fonction phi d'Euler: nombre d'entiers <= n premiers avec n.
    
    Args:
        n: Nombre
        
    Returns:
        phi(n)
        
    Example:
        >>> euler_phi(9)
        6
    """
    result = n
    p = 2
    
    while p * p <= n:
        if n % p == 0:
            while n % p == 0:
                n //= p
            result -= result // p
        p += 1
    
    if n > 1:
        result -= result // n
    
    return result


def euler_phi_sieve(n):
    """
    Calcule phi(i) pour tous i de 1 à n.
    
    Args:
        n: Limite supérieure
        
    Returns:
        Liste où phi[i] = phi(i)
    """
    phi = list(range(n + 1))
    
    for i in range(2, n + 1):
        if phi[i] == i:  # i est premier
            for j in range(i, n + 1, i):
                phi[j] -= phi[j] // i
    
    return phi


def test():
    """Tests unitaires complets"""
    
    # Test crible
    primes = sieve_of_eratosthenes(30)
    assert primes == [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
    
    # Test is_prime
    assert is_prime(2)
    assert is_prime(17)
    assert not is_prime(1)
    assert not is_prime(4)
    assert not is_prime(15)
    
    # Test factorisation
    factors = prime_factorization(60)
    assert factors == {2: 2, 3: 1, 5: 1}
    
    factors_list = prime_factorization_list(60)
    assert factors_list == [2, 2, 3, 5]
    
    # Test Miller-Rabin
    assert miller_rabin(17)
    assert miller_rabin(97)
    assert not miller_rabin(100)
    assert miller_rabin(1000000007)
    
    # Test comptage de diviseurs
    assert count_divisors(12) == 6  # 1, 2, 3, 4, 6, 12
    assert count_divisors(16) == 5  # 1, 2, 4, 8, 16
    
    # Test somme des diviseurs
    assert sum_of_divisors(12) == 28  # 1+2+3+4+6+12
    assert sum_of_divisors(6) == 12   # 1+2+3+6
    
    # Test Euler phi
    assert euler_phi(9) == 6  # 1,2,4,5,7,8 sont premiers avec 9
    assert euler_phi(10) == 4  # 1,3,7,9
    
    # Test SPF
    spf = smallest_prime_factor(20)
    assert spf[12] == 2
    assert spf[15] == 3
    assert spf[17] == 17  # 17 est premier
    
    # Test phi sieve
    phi_array = euler_phi_sieve(10)
    assert phi_array[9] == 6
    assert phi_array[10] == 4
    
    # Test edge cases
    assert sieve_of_eratosthenes(1) == []
    assert not is_prime(0)
    assert not is_prime(1)
    assert prime_factorization(1) == {}
    
    print("Tous les tests passes")


if __name__ == "__main__":
    test()

