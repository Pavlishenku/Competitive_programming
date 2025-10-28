"""
================================================================================
SUFFIX AUTOMATON
================================================================================

Description:
-----------
Automate des suffixes (Suffix Automaton) - structure compacte representant
tous les suffixes d'une chaine. Plus puissant que suffix array pour certains
problemes de pattern matching et de comptage.

Complexite:
-----------
- Construction: O(n * alphabet_size)
- Espace: O(n) etats
- Pattern matching: O(m) ou m = longueur pattern
- Comptage occurrences: O(m)

Cas d'usage typiques:
--------------------
1. Trouver toutes les occurrences d'un pattern
2. Compter les sous-chaines distinctes
3. Plus longue sous-chaine commune
4. k-ieme sous-chaine lexicographique

Problemes classiques:
--------------------
- Codeforces 235C - Cyclical Quest
- Codeforces 128B - String
- SPOJ SUBLEX - Lexicographical Substring Search
- AtCoder ABC 135F - Strings of Eternity

Auteur: Assistant CP
Date: 2025
================================================================================
"""

from typing import Dict, List, Optional


class SuffixAutomaton:
    """
    Suffix Automaton pour traitement de chaines.
    
    Exemple:
    --------
    >>> sa = SuffixAutomaton()
    >>> sa.build("abcbc")
    >>> 
    >>> # Verifie si pattern existe
    >>> print(sa.contains("bc"))  # True
    >>> print(sa.contains("cb"))  # True
    >>> print(sa.contains("xyz"))  # False
    >>> 
    >>> # Compte occurrences
    >>> print(sa.count_occurrences("bc"))  # 2
    >>> 
    >>> # Compte sous-chaines distinctes
    >>> print(sa.count_distinct_substrings())  # 13
    """
    
    class State:
        """Etat de l'automate"""
        
        def __init__(self):
            self.len = 0  # Longueur du plus long chemin
            self.link = -1  # Suffix link
            self.next: Dict[str, int] = {}  # Transitions
            self.cnt = 0  # Nombre d'occurrences (termine ici)
            self.first_pos = -1  # Premiere position d'occurrence
    
    def __init__(self):
        """Initialise l'automate vide"""
        self.states: List[SuffixAutomaton.State] = []
        self.last = 0  # Dernier etat
        
        # Etat initial
        initial = SuffixAutomaton.State()
        initial.len = 0
        initial.link = -1
        self.states.append(initial)
    
    def add_char(self, c: str):
        """
        Ajoute un caractere a l'automate.
        
        Time: O(alphabet_size) amorti
        
        Args:
            c: Caractere a ajouter
        """
        cur = len(self.states)
        new_state = SuffixAutomaton.State()
        new_state.len = self.states[self.last].len + 1
        new_state.first_pos = new_state.len - 1
        new_state.cnt = 1
        self.states.append(new_state)
        
        # Remonte les suffix links en ajoutant transition
        p = self.last
        while p != -1 and c not in self.states[p].next:
            self.states[p].next[c] = cur
            p = self.states[p].link
        
        if p == -1:
            new_state.link = 0
        else:
            q = self.states[p].next[c]
            if self.states[p].len + 1 == self.states[q].len:
                new_state.link = q
            else:
                # Clone l'etat q
                clone = len(self.states)
                cloned_state = SuffixAutomaton.State()
                cloned_state.len = self.states[p].len + 1
                cloned_state.next = self.states[q].next.copy()
                cloned_state.link = self.states[q].link
                cloned_state.first_pos = self.states[q].first_pos
                self.states.append(cloned_state)
                
                # Update links
                while p != -1 and self.states[p].next.get(c) == q:
                    self.states[p].next[c] = clone
                    p = self.states[p].link
                
                self.states[q].link = clone
                new_state.link = clone
        
        self.last = cur
    
    def build(self, s: str):
        """
        Construit l'automate pour une chaine.
        
        Time: O(n * alphabet_size)
        
        Args:
            s: Chaine a traiter
        """
        for c in s:
            self.add_char(c)
    
    def contains(self, pattern: str) -> bool:
        """
        Verifie si pattern est une sous-chaine.
        
        Time: O(len(pattern))
        
        Args:
            pattern: Pattern a chercher
            
        Returns:
            True si pattern existe dans la chaine
        """
        state = 0
        for c in pattern:
            if c not in self.states[state].next:
                return False
            state = self.states[state].next[c]
        return True
    
    def first_occurrence(self, pattern: str) -> int:
        """
        Trouve la premiere occurrence du pattern.
        
        Time: O(len(pattern))
        
        Args:
            pattern: Pattern a chercher
            
        Returns:
            Position de debut (-1 si non trouve)
        """
        state = 0
        for c in pattern:
            if c not in self.states[state].next:
                return -1
            state = self.states[state].next[c]
        
        # Position de fin - longueur + 1
        return self.states[state].first_pos - len(pattern) + 1
    
    def count_occurrences(self, pattern: str) -> int:
        """
        Compte le nombre d'occurrences d'un pattern.
        
        Time: O(len(pattern) + n) pour precalcul
        
        Args:
            pattern: Pattern a compter
            
        Returns:
            Nombre d'occurrences
        """
        # Precalcule cnt si necessaire
        if not hasattr(self, '_cnt_calculated'):
            self._calculate_cnt()
        
        state = 0
        for c in pattern:
            if c not in self.states[state].next:
                return 0
            state = self.states[state].next[c]
        
        return self.states[state].cnt
    
    def _calculate_cnt(self):
        """Precalcule le nombre d'occurrences pour chaque etat"""
        # Trie les etats par longueur decroissante
        order = sorted(range(len(self.states)), 
                      key=lambda i: self.states[i].len, 
                      reverse=True)
        
        for i in order:
            state = self.states[i]
            if state.link != -1:
                self.states[state.link].cnt += state.cnt
        
        self._cnt_calculated = True
    
    def count_distinct_substrings(self) -> int:
        """
        Compte le nombre de sous-chaines distinctes.
        
        Time: O(n)
        
        Returns:
            Nombre de sous-chaines distinctes
        """
        count = 0
        for i in range(1, len(self.states)):
            state = self.states[i]
            # Chaque etat apporte (len - link.len) nouvelles sous-chaines
            link_len = self.states[state.link].len if state.link != -1 else 0
            count += state.len - link_len
        
        return count
    
    def kth_substring(self, k: int) -> Optional[str]:
        """
        Trouve la k-ieme sous-chaine dans l'ordre lexicographique.
        
        Time: O(n^2) pire cas
        
        Args:
            k: Index (1-indexed)
            
        Returns:
            La k-ieme sous-chaine ou None si k trop grand
        """
        # Compte nombre de chemins depuis chaque etat
        dp = [0] * len(self.states)
        
        def dfs(v: int) -> int:
            if dp[v] != 0:
                return dp[v]
            
            dp[v] = 1  # Chaine vide
            for _, next_state in self.states[v].next.items():
                dp[v] += dfs(next_state)
            
            return dp[v]
        
        dfs(0)
        
        if k > dp[0] - 1:  # -1 car on ne compte pas la chaine vide
            return None
        
        # Construit la k-ieme chaine
        result = []
        state = 0
        
        while k > 0:
            # Trie les transitions par ordre lexicographique
            transitions = sorted(self.states[state].next.items())
            
            for c, next_state in transitions:
                cnt = dp[next_state]
                if k <= cnt:
                    result.append(c)
                    state = next_state
                    k -= 1
                    break
                k -= cnt
        
        return ''.join(result)
    
    def longest_common_substring(self, other_string: str) -> str:
        """
        Trouve la plus longue sous-chaine commune.
        
        Time: O(len(other_string))
        
        Args:
            other_string: Autre chaine a comparer
            
        Returns:
            Plus longue sous-chaine commune
        """
        state = 0
        length = 0
        max_length = 0
        max_pos = 0
        
        for i, c in enumerate(other_string):
            # Cherche transition avec c
            while state != 0 and c not in self.states[state].next:
                state = self.states[state].link
                length = self.states[state].len if state != 0 else 0
            
            if c in self.states[state].next:
                state = self.states[state].next[c]
                length += 1
            
            if length > max_length:
                max_length = length
                max_pos = i
        
        return other_string[max_pos - max_length + 1 : max_pos + 1]


def count_distinct_substrings(s: str) -> int:
    """
    Compte le nombre de sous-chaines distinctes dans s.
    
    Args:
        s: Chaine d'entree
    
    Returns:
        Nombre de sous-chaines distinctes
    
    Time: O(n * alphabet_size)
    """
    sa = SuffixAutomaton()
    sa.build(s)
    return sa.count_distinct_substrings()


def longest_common_substring(s1: str, s2: str) -> str:
    """
    Trouve la plus longue sous-chaine commune entre s1 et s2.
    
    Args:
        s1, s2: Chaines a comparer
    
    Returns:
        Plus longue sous-chaine commune
    
    Time: O(len(s1) + len(s2))
    """
    sa = SuffixAutomaton()
    sa.build(s1)
    return sa.longest_common_substring(s2)


# ============================================================================
# TESTS UNITAIRES
# ============================================================================

def test_suffix_automaton_basic():
    """Test basic suffix automaton"""
    sa = SuffixAutomaton()
    sa.build("abc")
    
    assert sa.contains("a")
    assert sa.contains("ab")
    assert sa.contains("abc")
    assert sa.contains("bc")
    assert sa.contains("c")
    assert not sa.contains("d")
    assert not sa.contains("ac")
    
    print("✓ Test suffix automaton basic passed")


def test_count_distinct():
    """Test comptage sous-chaines distinctes"""
    sa = SuffixAutomaton()
    sa.build("aaa")
    
    # "a", "aa", "aaa" = 3 distinctes
    assert sa.count_distinct_substrings() == 3
    
    sa2 = SuffixAutomaton()
    sa2.build("abc")
    
    # "a", "b", "c", "ab", "bc", "abc" = 6
    assert sa2.count_distinct_substrings() == 6
    
    print("✓ Test count distinct passed")


def test_count_occurrences():
    """Test comptage occurrences"""
    sa = SuffixAutomaton()
    sa.build("abcbc")
    
    assert sa.count_occurrences("bc") == 2
    assert sa.count_occurrences("ab") == 1
    assert sa.count_occurrences("c") == 2
    assert sa.count_occurrences("xyz") == 0
    
    print("✓ Test count occurrences passed")


def test_first_occurrence():
    """Test premiere occurrence"""
    sa = SuffixAutomaton()
    sa.build("abcabc")
    
    assert sa.first_occurrence("abc") == 0
    assert sa.first_occurrence("bc") == 1
    assert sa.first_occurrence("ca") == 2
    assert sa.first_occurrence("xyz") == -1
    
    print("✓ Test first occurrence passed")


def test_kth_substring():
    """Test k-ieme sous-chaine"""
    sa = SuffixAutomaton()
    sa.build("abc")
    
    # Ordre lex: "a", "ab", "abc", "b", "bc", "c"
    assert sa.kth_substring(1) == "a"
    assert sa.kth_substring(2) == "ab"
    assert sa.kth_substring(3) == "abc"
    assert sa.kth_substring(4) == "b"
    assert sa.kth_substring(5) == "bc"
    assert sa.kth_substring(6) == "c"
    
    print("✓ Test kth substring passed")


def test_longest_common_substring():
    """Test plus longue sous-chaine commune"""
    result = longest_common_substring("abcdef", "xbcdyz")
    assert result == "bcd"
    
    result2 = longest_common_substring("hello", "world")
    assert result2 in ["ll", "lo", "o"]  # Plusieurs reponses possibles
    
    print("✓ Test longest common substring passed")


def test_suffix_automaton_repeats():
    """Test avec repetitions"""
    sa = SuffixAutomaton()
    sa.build("aaaa")
    
    assert sa.contains("aa")
    assert sa.contains("aaa")
    assert sa.count_distinct_substrings() == 4  # "a", "aa", "aaa", "aaaa"
    
    print("✓ Test suffix automaton repeats passed")


# ============================================================================
# BENCHMARK
# ============================================================================

def benchmark_suffix_automaton():
    """Benchmark suffix automaton"""
    import time
    import random
    import string
    
    print("\n=== Benchmark Suffix Automaton ===")
    
    for n in [100, 1000, 5000, 10000]:
        # Genere chaine aleatoire
        s = ''.join(random.choices(string.ascii_lowercase[:5], k=n))
        
        # Construction
        start = time.time()
        sa = SuffixAutomaton()
        sa.build(s)
        build_time = time.time() - start
        
        # Count distinct
        start = time.time()
        count = sa.count_distinct_substrings()
        count_time = time.time() - start
        
        # Pattern matching
        patterns = [''.join(random.choices(string.ascii_lowercase[:5], k=5)) 
                   for _ in range(1000)]
        
        start = time.time()
        for p in patterns:
            sa.contains(p)
        search_time = time.time() - start
        
        print(f"\nn={n}:")
        print(f"  Build:   {build_time*1000:6.2f}ms")
        print(f"  Count:   {count_time*1000:6.2f}ms ({count} distinct)")
        print(f"  Search:  {search_time/len(patterns)*1000:6.3f}ms/query")


if __name__ == "__main__":
    # Tests
    test_suffix_automaton_basic()
    test_count_distinct()
    test_count_occurrences()
    test_first_occurrence()
    test_kth_substring()
    test_longest_common_substring()
    test_suffix_automaton_repeats()
    
    # Benchmark
    benchmark_suffix_automaton()
    
    print("\n✓ Tous les tests Suffix Automaton passes!")

