"""
Arithmétique Modulaire

Description:
    Opérations modulaires pour programmation compétitive:
    - Exponentiation rapide
    - Inverse modulaire
    - Théorème chinois des restes (CRT)
    - Calculs modulaires optimisés

Complexité:
    - Exponentiation: O(log n)
    - Inverse modulaire: O(log n)
    - CRT: O(n log n)

Cas d'usage:
    - Calculs avec grands nombres modulo p
    - Combinatoire modulaire
    - Cryptographie
    - Problèmes de congruences
    
Problèmes types:
    - Codeforces: 630I, 906D, 300C
    - AtCoder: ABC156D, ABC145D
    - CSES: Exponentiation, Binomial Coefficients
    
Implémentation par: 2025-10-27
Testé: Oui
"""


MOD = 10**9 + 7  # Module courant en CP


def pow_mod(base, exp, mod=MOD):
    """
    Exponentiation rapide modulaire: (base^exp) % mod
    
    Args:
        base: Base
        exp: Exposant
        mod: Module
        
    Returns:
        (base^exp) % mod
        
    Example:
        >>> pow_mod(2, 10, 1000)
        24
    """
    result = 1
    base = base % mod
    
    while exp > 0:
        if exp % 2 == 1:
            result = (result * base) % mod
        exp = exp >> 1
        base = (base * base) % mod
    
    return result


def mod_inverse(a, mod=MOD):
    """
    Inverse modulaire de a modulo mod (mod doit être premier).
    Utilise le théorème de Fermat: a^(p-1) ≡ 1 (mod p)
    Donc a^(-1) ≡ a^(p-2) (mod p)
    
    Args:
        a: Nombre
        mod: Module (doit être premier)
        
    Returns:
        Inverse modulaire de a
        
    Example:
        >>> mod_inverse(3, 7)
        5
    """
    return pow_mod(a, mod - 2, mod)


def extended_gcd(a, b):
    """
    PGCD étendu (algorithme d'Euclide étendu).
    Trouve x, y tels que ax + by = gcd(a, b)
    
    Args:
        a: Premier nombre
        b: Deuxième nombre
        
    Returns:
        Tuple (gcd, x, y)
        
    Example:
        >>> extended_gcd(30, 20)
        (10, 1, -1)
    """
    if b == 0:
        return (a, 1, 0)
    
    gcd, x1, y1 = extended_gcd(b, a % b)
    x = y1
    y = x1 - (a // b) * y1
    
    return (gcd, x, y)


def mod_inverse_extended(a, mod):
    """
    Inverse modulaire utilisant PGCD étendu.
    Fonctionne même si mod n'est pas premier (si gcd(a, mod) = 1).
    
    Args:
        a: Nombre
        mod: Module
        
    Returns:
        Inverse modulaire de a, ou None si n'existe pas
    """
    gcd, x, _ = extended_gcd(a, mod)
    
    if gcd != 1:
        return None  # Inverse n'existe pas
    
    return (x % mod + mod) % mod


def chinese_remainder_theorem(remainders, moduli):
    """
    Théorème chinois des restes.
    Résout le système: x ≡ r_i (mod m_i)
    
    Args:
        remainders: Liste des restes [r1, r2, ...]
        moduli: Liste des modules [m1, m2, ...] (doivent être premiers entre eux)
        
    Returns:
        Solution x (modulo prod(moduli))
        
    Example:
        >>> chinese_remainder_theorem([2, 3, 1], [3, 4, 5])
        11
    """
    total = 0
    prod = 1
    
    for m in moduli:
        prod *= m
    
    for r, m in zip(remainders, moduli):
        p = prod // m
        total += r * mod_inverse_extended(p, m) * p
    
    return total % prod


def factorial_mod(n, mod=MOD):
    """
    Calcule n! modulo mod.
    
    Args:
        n: Nombre
        mod: Module
        
    Returns:
        n! % mod
    """
    result = 1
    for i in range(2, n + 1):
        result = (result * i) % mod
    return result


class ModularFactorial:
    """
    Précalcule les factorielles et leurs inverses pour combinatoire rapide.
    """
    
    def __init__(self, max_n, mod=MOD):
        """
        Args:
            max_n: Taille maximale
            mod: Module (doit être premier)
        """
        self.mod = mod
        self.max_n = max_n
        
        # Précalculer factorielles
        self.fact = [1] * (max_n + 1)
        for i in range(1, max_n + 1):
            self.fact[i] = (self.fact[i-1] * i) % mod
        
        # Précalculer inverses des factorielles
        self.inv_fact = [1] * (max_n + 1)
        self.inv_fact[max_n] = mod_inverse(self.fact[max_n], mod)
        for i in range(max_n - 1, -1, -1):
            self.inv_fact[i] = (self.inv_fact[i+1] * (i+1)) % mod
    
    def comb(self, n, r):
        """
        Calcule C(n, r) = n! / (r! * (n-r)!) modulo mod.
        
        Args:
            n: Nombre total
            r: Nombre à choisir
            
        Returns:
            C(n, r) % mod
        """
        if r < 0 or r > n:
            return 0
        
        return (self.fact[n] * self.inv_fact[r] % self.mod 
                * self.inv_fact[n-r] % self.mod)
    
    def perm(self, n, r):
        """
        Calcule P(n, r) = n! / (n-r)! modulo mod.
        
        Args:
            n: Nombre total
            r: Nombre à arranger
            
        Returns:
            P(n, r) % mod
        """
        if r < 0 or r > n:
            return 0
        
        return (self.fact[n] * self.inv_fact[n-r]) % self.mod


def add_mod(a, b, mod=MOD):
    """Addition modulaire sécurisée"""
    return ((a % mod) + (b % mod)) % mod


def sub_mod(a, b, mod=MOD):
    """Soustraction modulaire sécurisée"""
    return ((a % mod) - (b % mod) + mod) % mod


def mul_mod(a, b, mod=MOD):
    """Multiplication modulaire sécurisée"""
    return ((a % mod) * (b % mod)) % mod


def div_mod(a, b, mod=MOD):
    """Division modulaire: a / b mod p"""
    return mul_mod(a, mod_inverse(b, mod), mod)


def test():
    """Tests unitaires complets"""
    
    # Test exponentiation
    assert pow_mod(2, 10, 1000) == 24
    assert pow_mod(3, 4, 17) == 13
    assert pow_mod(5, 0, 13) == 1
    
    # Test inverse modulaire
    assert mod_inverse(3, 7) == 5  # 3 * 5 = 15 ≡ 1 (mod 7)
    assert (3 * mod_inverse(3, 7)) % 7 == 1
    
    # Test PGCD étendu
    gcd, x, y = extended_gcd(30, 20)
    assert gcd == 10
    assert 30 * x + 20 * y == gcd
    
    # Test inverse modulaire étendu
    inv = mod_inverse_extended(3, 7)
    assert (3 * inv) % 7 == 1
    
    # Test CRT
    x = chinese_remainder_theorem([2, 3, 1], [3, 4, 5])
    assert x % 3 == 2
    assert x % 4 == 3
    assert x % 5 == 1
    
    # Test factorielle modulaire
    assert factorial_mod(5, 100) == 20  # 5! = 120 mod 100 = 20
    
    # Test ModularFactorial
    mod_fact = ModularFactorial(100)
    
    # C(5, 2) = 10
    assert mod_fact.comb(5, 2) == 10
    
    # C(10, 3) = 120
    assert mod_fact.comb(10, 3) == 120
    
    # P(5, 2) = 20
    assert mod_fact.perm(5, 2) == 20
    
    # Test opérations modulaires
    MOD_TEST = 13
    assert add_mod(10, 5, MOD_TEST) == 2
    assert sub_mod(3, 5, MOD_TEST) == 11
    assert mul_mod(5, 4, MOD_TEST) == 7
    assert div_mod(10, 2, MOD_TEST) == 5
    
    # Test edge cases
    assert pow_mod(0, 5, 7) == 0
    assert pow_mod(1, 1000, 7) == 1
    assert mod_fact.comb(5, 0) == 1
    assert mod_fact.comb(5, 5) == 1
    assert mod_fact.comb(5, 6) == 0
    
    # Test grand nombre
    large_mod = 10**9 + 7
    result = pow_mod(2, 1000000, large_mod)
    assert result > 0
    
    print("Tous les tests passes")


if __name__ == "__main__":
    test()

