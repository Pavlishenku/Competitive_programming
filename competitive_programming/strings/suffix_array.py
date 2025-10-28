"""
Suffix Array et LCP Array

Description:
    Structure pour manipuler tous les suffixes d'une chaîne triés.
    LCP (Longest Common Prefix) entre suffixes consécutifs.

Complexité:
    - Construction: O(n log n) avec SA-IS ou O(n log² n) avec sorting
    - LCP Array: O(n) avec algorithme de Kasai
    - Espace: O(n)

Cas d'usage:
    - Recherche de sous-chaînes
    - Plus longue sous-chaîne répétée
    - Nombre de sous-chaînes distinctes
    - Pattern matching
    
Problèmes types:
    - Codeforces: 432D, 873F
    - SPOJ: SUBST1, LCS2
    
Implémentation par: 2025-10-27
Testé: Oui
"""


class SuffixArray:
    """
    Suffix Array avec LCP Array.
    Construit en O(n log² n).
    """
    
    def __init__(self, s):
        """
        Args:
            s: Chaîne à analyser
            
        Example:
            >>> sa = SuffixArray("banana")
            >>> sa.suffix_array
            [5, 3, 1, 0, 4, 2]
        """
        self.s = s
        self.n = len(s)
        self.suffix_array = self._build_suffix_array()
        self.lcp = self._build_lcp_array()
    
    def _build_suffix_array(self):
        """
        Construit le suffix array en O(n log² n).
        """
        n = self.n
        suffixes = list(range(n))
        
        # Trier par premier caractère
        rank = [ord(c) for c in self.s]
        
        k = 1
        while k < n:
            # Trier par (rank[i], rank[i+k])
            def compare_key(i):
                return (rank[i], rank[i + k] if i + k < n else -1)
            
            suffixes.sort(key=compare_key)
            
            # Recalculer les ranks
            new_rank = [0] * n
            for i in range(1, n):
                new_rank[suffixes[i]] = new_rank[suffixes[i-1]]
                if compare_key(suffixes[i]) != compare_key(suffixes[i-1]):
                    new_rank[suffixes[i]] += 1
            
            rank = new_rank
            k *= 2
        
        return suffixes
    
    def _build_lcp_array(self):
        """
        Construit le LCP array en O(n) (algorithme de Kasai).
        lcp[i] = longueur du plus long préfixe commun entre
                 suffix_array[i] et suffix_array[i+1]
        """
        n = self.n
        lcp = [0] * n
        
        # Inverse du suffix array
        rank = [0] * n
        for i in range(n):
            rank[self.suffix_array[i]] = i
        
        h = 0
        for i in range(n):
            if rank[i] > 0:
                j = self.suffix_array[rank[i] - 1]
                
                while i + h < n and j + h < n and self.s[i + h] == self.s[j + h]:
                    h += 1
                
                lcp[rank[i]] = h
                
                if h > 0:
                    h -= 1
        
        return lcp
    
    def search(self, pattern):
        """
        Recherche un pattern dans la chaîne.
        
        Args:
            pattern: Motif à chercher
            
        Returns:
            Liste des positions où le pattern apparaît
        """
        positions = []
        
        # Binary search pour trouver la plage
        left, right = 0, self.n - 1
        
        # Trouver le premier suffixe >= pattern
        while left < right:
            mid = (left + right) // 2
            suffix = self.s[self.suffix_array[mid]:]
            
            if suffix < pattern:
                left = mid + 1
            else:
                right = mid
        
        # Vérifier tous les suffixes qui commencent par pattern
        i = left
        while i < self.n:
            pos = self.suffix_array[i]
            if self.s[pos:pos + len(pattern)] == pattern:
                positions.append(pos)
                i += 1
            else:
                break
        
        return sorted(positions)
    
    def longest_repeated_substring(self):
        """
        Trouve la plus longue sous-chaîne qui apparaît au moins 2 fois.
        
        Returns:
            Tuple (longueur, sous-chaîne)
        """
        if self.n == 0:
            return (0, "")
        
        max_lcp = max(self.lcp)
        
        if max_lcp == 0:
            return (0, "")
        
        # Trouver l'index avec max LCP
        idx = self.lcp.index(max_lcp)
        pos = self.suffix_array[idx]
        
        return (max_lcp, self.s[pos:pos + max_lcp])
    
    def count_distinct_substrings(self):
        """
        Compte le nombre de sous-chaînes distinctes.
        
        Returns:
            Nombre de sous-chaînes distinctes
        """
        # Total de sous-chaînes - doublons
        # Total = n*(n+1)/2
        # Doublons = sum(lcp)
        
        total = self.n * (self.n + 1) // 2
        duplicates = sum(self.lcp)
        
        return total - duplicates
    
    def longest_common_substring(self, other_string):
        """
        Trouve la plus longue sous-chaîne commune avec une autre chaîne.
        
        Args:
            other_string: Autre chaîne
            
        Returns:
            Longueur de la plus longue sous-chaîne commune
        """
        # Créer une chaîne combinée avec séparateur
        combined = self.s + "#" + other_string
        sa_combined = SuffixArray(combined)
        
        n1 = len(self.s)
        max_length = 0
        
        # Chercher le plus grand LCP entre suffixes de chaînes différentes
        for i in range(len(combined)):
            if sa_combined.lcp[i] > 0:
                pos1 = sa_combined.suffix_array[i - 1]
                pos2 = sa_combined.suffix_array[i]
                
                # Vérifier qu'ils viennent de chaînes différentes
                if (pos1 < n1) != (pos2 < n1):
                    max_length = max(max_length, sa_combined.lcp[i])
        
        return max_length


def kasai_lcp(s, suffix_array):
    """
    Algorithme de Kasai pour construire le LCP array.
    Version standalone.
    
    Args:
        s: Chaîne
        suffix_array: Suffix array de s
        
    Returns:
        LCP array
    """
    n = len(s)
    lcp = [0] * n
    rank = [0] * n
    
    for i in range(n):
        rank[suffix_array[i]] = i
    
    h = 0
    for i in range(n):
        if rank[i] > 0:
            j = suffix_array[rank[i] - 1]
            
            while i + h < n and j + h < n and s[i + h] == s[j + h]:
                h += 1
            
            lcp[rank[i]] = h
            
            if h > 0:
                h -= 1
    
    return lcp


def test():
    """Tests unitaires complets"""
    
    # Test suffix array
    sa = SuffixArray("banana")
    
    # Suffixes triés: "a", "ana", "anana", "banana", "na", "nana"
    # Positions:        5     3      1        0        4     2
    assert sa.suffix_array == [5, 3, 1, 0, 4, 2]
    
    # Test LCP
    # "a" vs "ana" -> 1
    # "ana" vs "anana" -> 3
    # etc.
    assert sa.lcp[0] == 0
    assert sa.lcp[1] == 1
    assert sa.lcp[2] == 3
    
    # Test search
    positions = sa.search("ana")
    assert 1 in positions
    assert 3 in positions
    
    # Test longest repeated substring
    length, substring = sa.longest_repeated_substring()
    assert length == 3
    assert substring == "ana"
    
    # Test distinct substrings
    # "banana" a 15 sous-chaînes distinctes
    distinct = sa.count_distinct_substrings()
    assert distinct == 15
    
    # Test avec chaîne simple
    sa2 = SuffixArray("aaa")
    length2, sub2 = sa2.longest_repeated_substring()
    assert length2 == 2
    assert sub2 == "aa"
    
    # Test count distinct
    distinct2 = sa2.count_distinct_substrings()
    assert distinct2 == 3  # "a", "aa", "aaa"
    
    # Test longest common substring
    sa3 = SuffixArray("abcde")
    lcs_len = sa3.longest_common_substring("cdefg")
    assert lcs_len == 3  # "cde"
    
    # Test search non trouvé
    positions_empty = sa.search("xyz")
    assert len(positions_empty) == 0
    
    # Test avec chaîne vide
    sa_empty = SuffixArray("")
    assert sa_empty.suffix_array == []
    assert sa_empty.count_distinct_substrings() == 0
    
    # Test kasai standalone
    lcp_kasai = kasai_lcp("banana", [5, 3, 1, 0, 4, 2])
    assert lcp_kasai == sa.lcp
    
    print("Tous les tests passes")


if __name__ == "__main__":
    test()

