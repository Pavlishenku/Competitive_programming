"""
FFT/NTT (Fast Fourier Transform / Number Theoretic Transform)

Description:
    FFT pour convolution rapide de polynômes ou tableaux.
    NTT est la version modulaire pour éviter les erreurs de précision.

Complexité:
    - FFT: O(n log n)
    - Convolution: O(n log n)

Cas d'usage:
    - Multiplication de grands nombres
    - Multiplication de polynômes
    - Convolution de tableaux
    - Problèmes de comptage
    
Problèmes types:
    - Codeforces: 528D, 993E
    - AtCoder: ABC196F
    
Implémentation par: 2025-10-27
Testé: Oui
"""

import math


def fft(a, inverse=False):
    """
    Fast Fourier Transform (version récursive).
    
    Args:
        a: Coefficients du polynôme (liste de complexes)
        inverse: Si True, effectue l'inverse FFT
        
    Returns:
        Transformée de Fourier
    """
    n = len(a)
    
    if n == 1:
        return a
    
    # Séparer en pairs et impairs
    even = fft([a[i] for i in range(0, n, 2)], inverse)
    odd = fft([a[i] for i in range(1, n, 2)], inverse)
    
    # Racine n-ième de l'unité
    angle = 2 * math.pi / n * (-1 if inverse else 1)
    w = complex(math.cos(angle), math.sin(angle))
    
    wn = complex(1, 0)
    result = [0] * n
    
    for i in range(n // 2):
        t = wn * odd[i]
        result[i] = even[i] + t
        result[i + n // 2] = even[i] - t
        wn *= w
    
    return result


def multiply_polynomials(p1, p2):
    """
    Multiplie deux polynômes en O(n log n).
    
    Args:
        p1: Coefficients du premier polynôme
        p2: Coefficients du deuxième polynôme
        
    Returns:
        Coefficients du produit
        
    Example:
        >>> p1 = [1, 2, 3]  # 1 + 2x + 3x²
        >>> p2 = [4, 5]     # 4 + 5x
        >>> multiply_polynomials(p1, p2)
        [4, 13, 22, 15]  # 4 + 13x + 22x² + 15x³
    """
    # Taille résultat
    result_size = len(p1) + len(p2) - 1
    
    # Arrondir à la prochaine puissance de 2
    n = 1
    while n < result_size:
        n *= 2
    
    # Padding avec des zéros
    a = p1 + [0] * (n - len(p1))
    b = p2 + [0] * (n - len(p2))
    
    # Convertir en complexes
    a = [complex(x, 0) for x in a]
    b = [complex(x, 0) for x in b]
    
    # FFT
    fa = fft(a)
    fb = fft(b)
    
    # Multiplication point par point
    fc = [fa[i] * fb[i] for i in range(n)]
    
    # Inverse FFT
    c = fft(fc, inverse=True)
    
    # Normaliser et arrondir
    result = [round(c[i].real / n) for i in range(result_size)]
    
    return result


def convolve(a, b):
    """
    Convolution de deux tableaux.
    
    Args:
        a: Premier tableau
        b: Deuxième tableau
        
    Returns:
        Convolution c où c[k] = sum(a[i] * b[k-i])
    """
    return multiply_polynomials(a, b)


class NTT:
    """
    Number Theoretic Transform (FFT modulaire).
    Plus précis que FFT car travaille avec des entiers.
    """
    
    def __init__(self, mod=998244353):
        """
        Args:
            mod: Module premier (doit avoir forme 2^k * q + 1)
                 998244353 = 119 * 2^23 + 1 (couramment utilisé)
        """
        self.mod = mod
        self.root = self._find_primitive_root()
    
    def _find_primitive_root(self):
        """Trouve une racine primitive modulo mod"""
        if self.mod == 998244353:
            return 3  # Racine primitive connue
        elif self.mod == 1000000007:
            return 5
        else:
            # Recherche simple (pas optimal)
            for g in range(2, self.mod):
                if self._is_primitive_root(g):
                    return g
        return 2
    
    def _is_primitive_root(self, g):
        """Vérifie si g est une racine primitive"""
        # Simplifié pour ce cas
        return pow(g, (self.mod - 1) // 2, self.mod) != 1
    
    def ntt(self, a, inverse=False):
        """
        Number Theoretic Transform.
        
        Args:
            a: Tableau d'entiers
            inverse: Si True, inverse NTT
            
        Returns:
            Transformée
        """
        n = len(a)
        a = a[:]
        
        # Bit reversal
        j = 0
        for i in range(1, n):
            bit = n >> 1
            while j & bit:
                j ^= bit
                bit >>= 1
            j ^= bit
            
            if i < j:
                a[i], a[j] = a[j], a[i]
        
        # NTT
        length = 2
        while length <= n:
            # Racine (2*length)-ième de l'unité
            if inverse:
                w = pow(self.root, (self.mod - 1) // length, self.mod)
                w = pow(w, self.mod - 2, self.mod)  # Inverse
            else:
                w = pow(self.root, (self.mod - 1) // length, self.mod)
            
            for i in range(0, n, length):
                wn = 1
                for j in range(length // 2):
                    u = a[i + j]
                    v = a[i + j + length // 2] * wn % self.mod
                    a[i + j] = (u + v) % self.mod
                    a[i + j + length // 2] = (u - v + self.mod) % self.mod
                    wn = wn * w % self.mod
            
            length *= 2
        
        if inverse:
            n_inv = pow(n, self.mod - 2, self.mod)
            a = [x * n_inv % self.mod for x in a]
        
        return a
    
    def multiply(self, p1, p2):
        """
        Multiplie deux polynômes modulo mod.
        
        Args:
            p1: Premier polynôme
            p2: Deuxième polynôme
            
        Returns:
            Produit modulo mod
        """
        result_size = len(p1) + len(p2) - 1
        
        n = 1
        while n < result_size:
            n *= 2
        
        a = p1 + [0] * (n - len(p1))
        b = p2 + [0] * (n - len(p2))
        
        fa = self.ntt(a)
        fb = self.ntt(b)
        
        fc = [(fa[i] * fb[i]) % self.mod for i in range(n)]
        
        c = self.ntt(fc, inverse=True)
        
        return c[:result_size]


def multiply_large_numbers(num1, num2, base=10):
    """
    Multiplie deux grands nombres en utilisant FFT.
    
    Args:
        num1: Premier nombre (string ou int)
        num2: Deuxième nombre (string ou int)
        base: Base numérique (10 par défaut)
        
    Returns:
        Produit (string)
    """
    s1 = str(num1)[::-1]  # Inverser pour avoir chiffres de poids faible en premier
    s2 = str(num2)[::-1]
    
    p1 = [int(c) for c in s1]
    p2 = [int(c) for c in s2]
    
    # Multiplier
    result = multiply_polynomials(p1, p2)
    
    # Gérer les retenues
    carry = 0
    for i in range(len(result)):
        result[i] += carry
        carry = result[i] // base
        result[i] %= base
    
    while carry:
        result.append(carry % base)
        carry //= base
    
    # Retirer les zéros de tête et inverser
    while len(result) > 1 and result[-1] == 0:
        result.pop()
    
    return ''.join(map(str, result[::-1]))


def test():
    """Tests unitaires complets"""
    
    # Test multiplication de polynômes
    p1 = [1, 2, 3]  # 1 + 2x + 3x²
    p2 = [4, 5]     # 4 + 5x
    product = multiply_polynomials(p1, p2)
    # (1 + 2x + 3x²)(4 + 5x) = 4 + 13x + 22x² + 15x³
    assert product == [4, 13, 22, 15]
    
    # Test convolution
    a = [1, 2, 3]
    b = [4, 5, 6]
    conv = convolve(a, b)
    assert len(conv) == 5
    
    # Test NTT
    ntt = NTT(mod=998244353)
    p1_ntt = [1, 2, 3, 4]
    p2_ntt = [5, 6, 7]
    product_ntt = ntt.multiply(p1_ntt, p2_ntt)
    
    # Vérifier avec multiplication directe
    expected = multiply_polynomials(p1_ntt, p2_ntt)
    for i in range(len(expected)):
        assert product_ntt[i] == expected[i] % ntt.mod
    
    # Test multiplication de grands nombres
    num1 = 12345
    num2 = 67890
    result = multiply_large_numbers(num1, num2)
    assert int(result) == 12345 * 67890
    
    # Test avec des nombres très grands
    big1 = "123456789012345678901234567890"
    big2 = "987654321098765432109876543210"
    result_big = multiply_large_numbers(big1, big2)
    assert int(result_big) == int(big1) * int(big2)
    
    # Test edge cases
    assert multiply_polynomials([1], [1]) == [1]
    assert multiply_polynomials([0], [5]) == [0]
    
    print("Tous les tests passes")


if __name__ == "__main__":
    test()

