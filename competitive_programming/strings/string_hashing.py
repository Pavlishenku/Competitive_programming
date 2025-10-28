"""
String Hashing

Description:
    Techniques de hachage de chaînes pour comparaisons rapides.
    Utilise polynomial rolling hash et double hashing.

Complexité:
    - Preprocessing: O(n)
    - Query: O(1)
    - Espace: O(n)

Cas d'usage:
    - Comparaison de sous-chaînes en O(1)
    - Recherche de motifs multiples
    - Détection de sous-chaînes communes
    - Hachage robuste contre collisions
    
Problèmes types:
    - Codeforces: 127E, 154C, 825F
    - AtCoder: ABC141E, ABC135D
    - CSES: String Hashing
    
Implémentation par: 2025-10-27
Testé: Oui
"""


class StringHash:
    """
    Single hash pour comparaison rapide de sous-chaînes.
    """
    
    def __init__(self, s, base=31, mod=10**9 + 7):
        """
        Args:
            s: Chaîne à hacher
            base: Base pour polynomial hash
            mod: Module pour éviter overflow
        """
        self.s = s
        self.n = len(s)
        self.base = base
        self.mod = mod
        
        # Précalculer les hashs préfixes
        self.prefix_hash = [0] * (self.n + 1)
        self.power = [1] * (self.n + 1)
        
        for i in range(self.n):
            self.prefix_hash[i + 1] = (self.prefix_hash[i] * base + ord(s[i])) % mod
            self.power[i + 1] = (self.power[i] * base) % mod
    
    def get_hash(self, left, right):
        """
        Hash de s[left:right].
        
        Args:
            left: Début (inclus)
            right: Fin (exclus)
            
        Returns:
            Hash de la sous-chaîne
        """
        result = (self.prefix_hash[right] - 
                 self.prefix_hash[left] * self.power[right - left]) % self.mod
        return (result + self.mod) % self.mod
    
    def compare(self, l1, r1, l2, r2):
        """Compare deux sous-chaînes par hash"""
        if r1 - l1 != r2 - l2:
            return False
        return self.get_hash(l1, r1) == self.get_hash(l2, r2)


class DoubleHash:
    """
    Double hashing pour réduire risque de collisions.
    Utilise deux modules différents.
    """
    
    def __init__(self, s, base1=31, base2=37, 
                 mod1=10**9 + 7, mod2=10**9 + 9):
        """
        Args:
            s: Chaîne à hacher
            base1, base2: Bases pour les deux hashs
            mod1, mod2: Modules pour les deux hashs
        """
        self.s = s
        self.n = len(s)
        
        # Premier hash
        self.base1 = base1
        self.mod1 = mod1
        self.hash1 = [0] * (self.n + 1)
        self.power1 = [1] * (self.n + 1)
        
        # Deuxième hash
        self.base2 = base2
        self.mod2 = mod2
        self.hash2 = [0] * (self.n + 1)
        self.power2 = [1] * (self.n + 1)
        
        for i in range(self.n):
            # Hash 1
            self.hash1[i + 1] = (self.hash1[i] * base1 + ord(s[i])) % mod1
            self.power1[i + 1] = (self.power1[i] * base1) % mod1
            
            # Hash 2
            self.hash2[i + 1] = (self.hash2[i] * base2 + ord(s[i])) % mod2
            self.power2[i + 1] = (self.power2[i] * base2) % mod2
    
    def get_hash(self, left, right):
        """
        Retourne le double hash de s[left:right].
        
        Args:
            left: Début
            right: Fin
            
        Returns:
            Tuple (hash1, hash2)
        """
        h1 = (self.hash1[right] - 
              self.hash1[left] * self.power1[right - left]) % self.mod1
        h1 = (h1 + self.mod1) % self.mod1
        
        h2 = (self.hash2[right] - 
              self.hash2[left] * self.power2[right - left]) % self.mod2
        h2 = (h2 + self.mod2) % self.mod2
        
        return (h1, h2)
    
    def compare(self, l1, r1, l2, r2):
        """Compare deux sous-chaînes avec double hash"""
        if r1 - l1 != r2 - l2:
            return False
        return self.get_hash(l1, r1) == self.get_hash(l2, r2)


def longest_common_substring(s1, s2):
    """
    Trouve la plus longue sous-chaîne commune.
    Utilise binary search + hashing.
    
    Args:
        s1: Première chaîne
        s2: Deuxième chaîne
        
    Returns:
        Longueur de la plus longue sous-chaîne commune
    """
    def check(length):
        """Vérifie s'il existe une sous-chaîne commune de longueur length"""
        if length == 0:
            return True
        
        # Hashs de toutes les sous-chaînes de s1
        hash1 = DoubleHash(s1)
        hashes = set()
        for i in range(len(s1) - length + 1):
            hashes.add(hash1.get_hash(i, i + length))
        
        # Vérifier dans s2
        hash2 = DoubleHash(s2)
        for i in range(len(s2) - length + 1):
            if hash2.get_hash(i, i + length) in hashes:
                return True
        
        return False
    
    # Binary search sur la longueur
    left, right = 0, min(len(s1), len(s2))
    result = 0
    
    while left <= right:
        mid = (left + right) // 2
        if check(mid):
            result = mid
            left = mid + 1
        else:
            right = mid - 1
    
    return result


def count_distinct_substrings(s):
    """
    Compte le nombre de sous-chaînes distinctes.
    
    Args:
        s: Chaîne
        
    Returns:
        Nombre de sous-chaînes distinctes
    """
    n = len(s)
    hash_obj = DoubleHash(s)
    
    unique_hashes = set()
    
    for length in range(1, n + 1):
        for start in range(n - length + 1):
            h = hash_obj.get_hash(start, start + length)
            unique_hashes.add(h)
    
    return len(unique_hashes)


def find_all_palindromes_hash(s):
    """
    Trouve tous les palindromes en utilisant hashing.
    
    Args:
        s: Chaîne
        
    Returns:
        Set de tous les palindromes
    """
    n = len(s)
    s_rev = s[::-1]
    
    hash_forward = DoubleHash(s)
    hash_backward = DoubleHash(s_rev)
    
    palindromes = set()
    
    for i in range(n):
        for j in range(i + 1, n + 1):
            # Comparer s[i:j] avec son reverse
            forward_hash = hash_forward.get_hash(i, j)
            # Le reverse de s[i:j] est s_rev[n-j:n-i]
            backward_hash = hash_backward.get_hash(n - j, n - i)
            
            if forward_hash == backward_hash:
                palindromes.add(s[i:j])
    
    return palindromes


def repeated_substring(s, k):
    """
    Trouve le plus long sous-chaîne apparaissant au moins k fois.
    
    Args:
        s: Chaîne
        k: Nombre minimum d'occurrences
        
    Returns:
        Longueur du plus long sous-chaîne répété k fois
    """
    def check(length):
        """Vérifie s'il existe une sous-chaîne de longueur length répétée k fois"""
        if length == 0:
            return True
        
        hash_obj = DoubleHash(s)
        hash_count = {}
        
        for i in range(len(s) - length + 1):
            h = hash_obj.get_hash(i, i + length)
            hash_count[h] = hash_count.get(h, 0) + 1
            if hash_count[h] >= k:
                return True
        
        return False
    
    # Binary search
    left, right = 0, len(s)
    result = 0
    
    while left <= right:
        mid = (left + right) // 2
        if check(mid):
            result = mid
            left = mid + 1
        else:
            right = mid - 1
    
    return result


def test():
    """Tests unitaires complets"""
    
    # Test StringHash
    s = "abcabc"
    sh = StringHash(s)
    
    # "abc" apparaît à 0 et 3
    h1 = sh.get_hash(0, 3)
    h2 = sh.get_hash(3, 6)
    assert h1 == h2
    
    # Test compare
    assert sh.compare(0, 3, 3, 6)
    assert not sh.compare(0, 2, 3, 6)
    
    # Test DoubleHash
    dh = DoubleHash("abcdefgh")
    
    # "abc" != "def"
    h1 = dh.get_hash(0, 3)
    h2 = dh.get_hash(3, 6)
    assert h1 != h2
    
    # "abc" == "abc"
    assert dh.compare(0, 3, 0, 3)
    
    # Test longest common substring
    lcs = longest_common_substring("abcde", "cdeab")
    assert lcs == 3  # "cde"
    
    lcs2 = longest_common_substring("abc", "xyz")
    assert lcs2 == 0
    
    # Test count distinct substrings
    count = count_distinct_substrings("aaa")
    # "a", "aa", "aaa"
    assert count == 3
    
    # Test repeated substring
    rep = repeated_substring("abcabcabc", 3)
    assert rep == 3  # "abc" apparaît 3 fois
    
    rep2 = repeated_substring("aaaa", 2)
    assert rep2 >= 3  # "aaa" apparaît 2 fois
    
    # Test edge cases
    sh_empty = StringHash("")
    assert sh_empty.get_hash(0, 0) == 0
    
    # Test palindrome detection avec hash
    palindromes = find_all_palindromes_hash("aba")
    assert "a" in palindromes
    assert "aba" in palindromes
    
    print("Tous les tests passes")


if __name__ == "__main__":
    test()

