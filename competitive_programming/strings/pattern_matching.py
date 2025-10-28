"""
Pattern Matching (KMP, Z-Algorithm, Rabin-Karp)

Description:
    Algorithmes de recherche de motifs dans des chaînes:
    - KMP (Knuth-Morris-Pratt)
    - Z-Algorithm
    - Rabin-Karp (rolling hash)

Complexité:
    - KMP: O(n + m) préprocessing + recherche
    - Z-Algorithm: O(n)
    - Rabin-Karp: O(n + m) en moyenne, O(nm) pire cas

Cas d'usage:
    - Recherche de sous-chaînes
    - Pattern matching multiple
    - Détection de périodicité
    - Comparaison de chaînes
    
Problèmes types:
    - Codeforces: 126B, 432D, 535D
    - AtCoder: ABC141E, ABC150F
    - CSES: String Matching, Finding Patterns
    
Implémentation par: 2025-10-27
Testé: Oui
"""


def compute_lps(pattern):
    """
    Calcule le tableau LPS (Longest Proper Prefix which is also Suffix) pour KMP.
    
    Args:
        pattern: Motif à analyser
        
    Returns:
        Tableau LPS
    """
    m = len(pattern)
    lps = [0] * m
    length = 0
    i = 1
    
    while i < m:
        if pattern[i] == pattern[length]:
            length += 1
            lps[i] = length
            i += 1
        else:
            if length != 0:
                length = lps[length - 1]
            else:
                lps[i] = 0
                i += 1
    
    return lps


def kmp_search(text, pattern):
    """
    Algorithme KMP pour rechercher toutes les occurrences d'un motif.
    
    Args:
        text: Texte dans lequel chercher
        pattern: Motif à chercher
        
    Returns:
        Liste des indices de début des occurrences
        
    Example:
        >>> kmp_search("ababcababa", "aba")
        [0, 5, 7]
    """
    n = len(text)
    m = len(pattern)
    
    if m == 0:
        return []
    
    lps = compute_lps(pattern)
    result = []
    
    i = 0  # index pour text
    j = 0  # index pour pattern
    
    while i < n:
        if text[i] == pattern[j]:
            i += 1
            j += 1
        
        if j == m:
            result.append(i - j)
            j = lps[j - 1]
        elif i < n and text[i] != pattern[j]:
            if j != 0:
                j = lps[j - 1]
            else:
                i += 1
    
    return result


def z_algorithm(s):
    """
    Calcule le Z-array: z[i] = longueur du plus long préfixe commun
    entre s et s[i:].
    
    Args:
        s: Chaîne à analyser
        
    Returns:
        Z-array
        
    Example:
        >>> z_algorithm("aabcaabxaaz")
        [11, 1, 0, 0, 3, 1, 0, 0, 2, 1, 0]
    """
    n = len(s)
    z = [0] * n
    z[0] = n
    
    l, r = 0, 0
    for i in range(1, n):
        if i <= r:
            z[i] = min(r - i + 1, z[i - l])
        
        while i + z[i] < n and s[z[i]] == s[i + z[i]]:
            z[i] += 1
        
        if i + z[i] - 1 > r:
            l, r = i, i + z[i] - 1
    
    return z


def z_search(text, pattern):
    """
    Utilise Z-algorithm pour rechercher un motif.
    
    Args:
        text: Texte dans lequel chercher
        pattern: Motif à chercher
        
    Returns:
        Liste des indices des occurrences
    """
    combined = pattern + "$" + text
    z = z_algorithm(combined)
    
    m = len(pattern)
    result = []
    
    for i in range(m + 1, len(combined)):
        if z[i] == m:
            result.append(i - m - 1)
    
    return result


class RollingHash:
    """
    Rolling Hash pour comparaison rapide de sous-chaînes.
    """
    
    def __init__(self, s, base=31, mod=10**9 + 7):
        """
        Args:
            s: Chaîne à hacher
            base: Base pour le hachage
            mod: Module
        """
        self.n = len(s)
        self.base = base
        self.mod = mod
        
        # Précalculer les hashs et puissances
        self.hash = [0] * (self.n + 1)
        self.power = [1] * (self.n + 1)
        
        for i in range(self.n):
            self.hash[i + 1] = (self.hash[i] * base + ord(s[i])) % mod
            self.power[i + 1] = (self.power[i] * base) % mod
    
    def get_hash(self, left, right):
        """
        Hash de la sous-chaîne [left, right).
        
        Args:
            left: Début (inclus)
            right: Fin (exclus)
            
        Returns:
            Hash de s[left:right]
        """
        result = (self.hash[right] - self.hash[left] * self.power[right - left]) % self.mod
        return (result + self.mod) % self.mod
    
    def compare(self, l1, r1, l2, r2):
        """
        Compare deux sous-chaînes.
        
        Args:
            l1, r1: Première sous-chaîne [l1, r1)
            l2, r2: Deuxième sous-chaîne [l2, r2)
            
        Returns:
            True si identiques (avec haute probabilité)
        """
        if r1 - l1 != r2 - l2:
            return False
        return self.get_hash(l1, r1) == self.get_hash(l2, r2)


def rabin_karp(text, pattern, base=256, mod=10**9 + 7):
    """
    Algorithme de Rabin-Karp avec rolling hash.
    
    Args:
        text: Texte dans lequel chercher
        pattern: Motif à chercher
        base: Base pour le hachage
        mod: Module
        
    Returns:
        Liste des indices des occurrences
    """
    n = len(text)
    m = len(pattern)
    
    if m > n:
        return []
    
    # Hash du pattern
    pattern_hash = 0
    for c in pattern:
        pattern_hash = (pattern_hash * base + ord(c)) % mod
    
    # Hash de la première fenêtre
    window_hash = 0
    for i in range(m):
        window_hash = (window_hash * base + ord(text[i])) % mod
    
    # Puissance pour rolling hash
    h = pow(base, m - 1, mod)
    
    result = []
    
    for i in range(n - m + 1):
        if window_hash == pattern_hash:
            # Vérification caractère par caractère pour éviter collisions
            if text[i:i+m] == pattern:
                result.append(i)
        
        # Rolling hash pour fenêtre suivante
        if i < n - m:
            window_hash = (window_hash - ord(text[i]) * h) % mod
            window_hash = (window_hash * base + ord(text[i + m])) % mod
            window_hash = (window_hash + mod) % mod
    
    return result


def manacher(s):
    """
    Algorithme de Manacher pour trouver tous les palindromes.
    
    Args:
        s: Chaîne à analyser
        
    Returns:
        Tuple (longest_palindrome_length, center, radius_array)
    """
    # Transformer la chaîne: "abc" -> "#a#b#c#"
    t = '#'.join('^{}$'.format(s))
    n = len(t)
    p = [0] * n  # p[i] = rayon du palindrome centré en i
    
    center = 0
    right = 0
    
    for i in range(1, n - 1):
        if i < right:
            mirror = 2 * center - i
            p[i] = min(right - i, p[mirror])
        
        # Expansion
        while t[i + p[i] + 1] == t[i - p[i] - 1]:
            p[i] += 1
        
        # Mettre à jour center et right
        if i + p[i] > right:
            center = i
            right = i + p[i]
    
    # Trouver le plus long palindrome
    max_len = 0
    center_index = 0
    for i in range(n):
        if p[i] > max_len:
            max_len = p[i]
            center_index = i
    
    return (max_len, center_index, p)


def longest_palindrome(s):
    """
    Trouve le plus long sous-palindrome.
    
    Args:
        s: Chaîne
        
    Returns:
        Plus long palindrome
    """
    if not s:
        return ""
    
    max_len, center_index, _ = manacher(s)
    
    # Extraire le palindrome
    start = (center_index - max_len) // 2
    return s[start:start + max_len]


def test():
    """Tests unitaires complets"""
    
    # Test KMP
    text = "ababcababa"
    pattern = "aba"
    matches = kmp_search(text, pattern)
    assert matches == [0, 5, 7]
    
    # Test Z-algorithm
    z = z_algorithm("aabcaabxaaz")
    assert z[0] == 11
    assert z[4] == 3
    
    # Test Z-search
    matches_z = z_search("ababcababa", "aba")
    assert matches_z == [0, 5, 7]
    
    # Test Rolling Hash
    s = "abcdefgh"
    rh = RollingHash(s)
    
    # "abc" == s[0:3]
    h1 = rh.get_hash(0, 3)
    # "def" == s[3:6]
    h2 = rh.get_hash(3, 6)
    assert h1 != h2
    
    # Comparer "abc" avec "abc"
    assert rh.compare(0, 3, 0, 3)
    
    # Test Rabin-Karp
    matches_rk = rabin_karp("ababcababa", "aba")
    assert matches_rk == [0, 5, 7]
    
    # Test Manacher
    max_len, _, _ = manacher("abacabad")
    assert max_len > 0
    
    # Test longest palindrome
    palindrome = longest_palindrome("babad")
    assert palindrome in ["bab", "aba"]
    
    palindrome2 = longest_palindrome("cbbd")
    assert palindrome2 == "bb"
    
    # Test edge cases
    assert kmp_search("", "a") == []
    assert kmp_search("a", "") == []
    assert kmp_search("aaa", "a") == [0, 1, 2]
    
    # Test KMP avec pattern non trouvé
    assert kmp_search("abcdef", "xyz") == []
    
    # Test Z-algorithm sur chaîne simple
    z_simple = z_algorithm("aaa")
    assert z_simple == [3, 2, 1]
    
    print("Tous les tests passes")


if __name__ == "__main__":
    test()

