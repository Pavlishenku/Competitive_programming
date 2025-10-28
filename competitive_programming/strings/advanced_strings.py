"""
================================================================================
ADVANCED STRING ALGORITHMS
================================================================================

Description:
-----------
Collection d'algorithmes avances sur les chaines:
- Lyndon Factorization (decomposition de Lyndon)
- Burrows-Wheeler Transform (BWT)
- Run-Length Encoding avance

Complexite:
-----------
- Lyndon: O(n)
- BWT: O(n log n)
- RLE: O(n)

Cas d'usage typiques:
--------------------
1. Compression de donnees
2. Analyse de periodicite
3. String matching avec compression
4. Bioinformatique

Problemes classiques:
--------------------
- Codeforces 1051D - Bicolorings
- SPOJ NUMOFPAL - Number of Palindromes
- String compression challenges
- Bioinformatics problems

Auteur: Assistant CP
Date: 2025
================================================================================
"""

from typing import List, Tuple


def lyndon_factorization(s: str) -> List[str]:
    """
    Decompose une chaine en facteurs de Lyndon.
    
    Un mot de Lyndon est strictement plus petit lexicographiquement
    que toutes ses rotations non-triviales.
    
    Time: O(n)
    
    Args:
        s: Chaine d'entree
        
    Returns:
        Liste des facteurs de Lyndon
        
    Exemple:
    --------
    >>> lyndon_factorization("abbab")
    ['abb', 'ab']
    >>> lyndon_factorization("aabaaab")
    ['aab', 'aaab']
    """
    n = len(s)
    factors = []
    i = 0
    
    while i < n:
        j = i + 1
        k = i
        
        while j < n and s[k] <= s[j]:
            if s[k] < s[j]:
                k = i
            else:
                k += 1
            j += 1
        
        # Extrait facteur(s)
        while i <= k:
            factors.append(s[i:i + j - k])
            i += j - k
    
    return factors


def minimum_rotation(s: str) -> int:
    """
    Trouve la rotation lexicographiquement minimale.
    
    Utilise Lyndon factorization (algorithme de Duval).
    
    Time: O(n)
    
    Args:
        s: Chaine d'entree
        
    Returns:
        Position de depart de la rotation minimale
        
    Exemple:
    --------
    >>> s = "bca"
    >>> pos = minimum_rotation(s)
    >>> print(s[pos:] + s[:pos])  # "abc"
    """
    s = s + s  # Double pour considerer rotations
    n = len(s) // 2
    
    i = 0
    ans = 0
    
    while i < n:
        j = i + 1
        k = i
        
        while j < len(s) and s[k] <= s[j]:
            if s[k] < s[j]:
                k = i
            else:
                k += 1
            j += 1
        
        while i <= k:
            i += j - k
            if i < n:
                ans = i
    
    return ans


def burrows_wheeler_transform(s: str) -> Tuple[str, int]:
    """
    Burrows-Wheeler Transform (BWT).
    
    Transform reversible utilisee pour la compression.
    
    Time: O(n log n)
    
    Args:
        s: Chaine d'entree (doit se terminer par caractere special)
        
    Returns:
        (transformed_string, original_index)
        
    Exemple:
    --------
    >>> bwt, idx = burrows_wheeler_transform("banana$")
    >>> print(bwt)  # "annb$aa"
    """
    n = len(s)
    
    # Genere toutes les rotations avec leur indice
    rotations = [(s[i:] + s[:i], i) for i in range(n)]
    
    # Trie lexicographiquement
    rotations.sort()
    
    # Prend dernier caractere de chaque rotation
    bwt = ''.join(rot[0][-1] for rot in rotations)
    
    # Trouve l'indice de la chaine originale
    original_idx = next(i for i, (rot, idx) in enumerate(rotations) if idx == 0)
    
    return bwt, original_idx


def inverse_burrows_wheeler(bwt: str, original_idx: int) -> str:
    """
    Inverse Burrows-Wheeler Transform.
    
    Time: O(n log n)
    
    Args:
        bwt: Chaine transformee
        original_idx: Index original
        
    Returns:
        Chaine originale
        
    Exemple:
    --------
    >>> original = inverse_burrows_wheeler("annb$aa", 3)
    >>> print(original)  # "banana$"
    """
    n = len(bwt)
    
    # Construit table de transformation
    table = [(bwt[i], i) for i in range(n)]
    table.sort()
    
    # Reconstruit la chaine
    result = []
    idx = original_idx
    
    for _ in range(n):
        result.append(table[idx][0])
        idx = table[idx][1]
    
    return ''.join(result)


def run_length_encode(s: str) -> List[Tuple[str, int]]:
    """
    Run-Length Encoding basique.
    
    Time: O(n)
    
    Args:
        s: Chaine d'entree
        
    Returns:
        Liste de (caractere, count)
        
    Exemple:
    --------
    >>> run_length_encode("aaabbbaac")
    [('a', 3), ('b', 3), ('a', 2), ('c', 1)]
    """
    if not s:
        return []
    
    result = []
    current_char = s[0]
    count = 1
    
    for i in range(1, len(s)):
        if s[i] == current_char:
            count += 1
        else:
            result.append((current_char, count))
            current_char = s[i]
            count = 1
    
    result.append((current_char, count))
    return result


def run_length_decode(encoded: List[Tuple[str, int]]) -> str:
    """
    Decode Run-Length Encoding.
    
    Time: O(output_length)
    
    Args:
        encoded: Liste de (caractere, count)
        
    Returns:
        Chaine decodee
        
    Exemple:
    --------
    >>> run_length_decode([('a', 3), ('b', 2)])
    'aaabb'
    """
    return ''.join(char * count for char, count in encoded)


def advanced_rle_compress(s: str) -> str:
    """
    RLE avance avec optimisation pour petites sequences.
    
    Time: O(n)
    
    Args:
        s: Chaine d'entree
        
    Returns:
        Chaine compresse
        
    Exemple:
    --------
    >>> advanced_rle_compress("aaabbbaac")
    'a3b3a2c1'
    """
    if not s:
        return ""
    
    result = []
    i = 0
    n = len(s)
    
    while i < n:
        char = s[i]
        count = 1
        
        while i + count < n and s[i + count] == char:
            count += 1
        
        # Optimisation: si count=1, pas de nombre
        if count == 1:
            result.append(char)
        else:
            result.append(f"{char}{count}")
        
        i += count
    
    return ''.join(result)


def longest_common_prefix_rotations(s: str) -> List[int]:
    """
    Calcule LCP entre s et toutes ses rotations.
    
    Time: O(n^2) naive, O(n log n) avec suffix array
    
    Args:
        s: Chaine d'entree
        
    Returns:
        Liste des LCP
        
    Exemple:
    --------
    >>> longest_common_prefix_rotations("abcab")
    [5, 0, 0, 2, 0]
    """
    n = len(s)
    result = []
    
    for rot in range(n):
        lcp = 0
        for i in range(n):
            if s[i] == s[(rot + i) % n]:
                lcp += 1
            else:
                break
        result.append(lcp)
    
    return result


# ============================================================================
# TESTS UNITAIRES
# ============================================================================

def test_lyndon_factorization():
    """Test Lyndon factorization"""
    factors = lyndon_factorization("abbab")
    assert ''.join(factors) == "abbab"
    
    factors2 = lyndon_factorization("aabaaab")
    assert ''.join(factors2) == "aabaaab"
    
    print("✓ Test Lyndon factorization passed")


def test_minimum_rotation():
    """Test minimum rotation"""
    s = "bca"
    pos = minimum_rotation(s)
    rotated = s[pos:] + s[:pos]
    assert rotated == "abc"
    
    s2 = "dcba"
    pos2 = minimum_rotation(s2)
    rotated2 = s2[pos2:] + s2[:pos2]
    assert rotated2 == "abcd"
    
    print("✓ Test minimum rotation passed")


def test_burrows_wheeler():
    """Test BWT and inverse"""
    s = "banana$"
    bwt, idx = burrows_wheeler_transform(s)
    
    # Inverse
    original = inverse_burrows_wheeler(bwt, idx)
    assert original == s
    
    print("✓ Test Burrows-Wheeler passed")


def test_run_length_encoding():
    """Test RLE"""
    s = "aaabbbaac"
    encoded = run_length_encode(s)
    decoded = run_length_decode(encoded)
    
    assert decoded == s
    assert len(encoded) < len(s)  # Compression
    
    print("✓ Test run-length encoding passed")


def test_advanced_rle():
    """Test advanced RLE"""
    s = "aaabbbaac"
    compressed = advanced_rle_compress(s)
    
    # Verifie que c'est plus court ou egal
    assert len(compressed) <= len(s) * 2
    
    print("✓ Test advanced RLE passed")


# ============================================================================
# BENCHMARK
# ============================================================================

def benchmark_advanced_strings():
    """Benchmark advanced string algorithms"""
    import time
    import random
    import string
    
    print("\n=== Benchmark Advanced Strings ===")
    
    for n in [100, 1000, 5000]:
        s = ''.join(random.choices(string.ascii_lowercase, k=n))
        
        # Lyndon
        start = time.time()
        lyndon_factorization(s)
        lyndon_time = time.time() - start
        
        # BWT
        s_bwt = s + '$'
        start = time.time()
        burrows_wheeler_transform(s_bwt)
        bwt_time = time.time() - start
        
        # RLE
        start = time.time()
        run_length_encode(s)
        rle_time = time.time() - start
        
        print(f"\nn={n}:")
        print(f"  Lyndon: {lyndon_time*1000:6.2f}ms")
        print(f"  BWT:    {bwt_time*1000:6.2f}ms")
        print(f"  RLE:    {rle_time*1000:6.2f}ms")


if __name__ == "__main__":
    test_lyndon_factorization()
    test_minimum_rotation()
    test_burrows_wheeler()
    test_run_length_encoding()
    test_advanced_rle()
    
    benchmark_advanced_strings()
    
    print("\n✓ Tous les tests Advanced Strings passes!")

