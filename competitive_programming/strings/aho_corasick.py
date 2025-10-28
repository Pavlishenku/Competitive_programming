"""
Aho-Corasick Algorithm

Description:
    Algorithme pour rechercher plusieurs motifs simultanément dans un texte.
    Construit un automate à partir d'un Trie avec des liens de défaillance.

Complexité:
    - Construction: O(sum(|pattern_i|))
    - Recherche: O(|text| + |output|)
    - Espace: O(sum(|pattern_i|) * |alphabet|)

Cas d'usage:
    - Recherche de multiples patterns
    - Détection de mots interdits
    - Analyse de texte avec dictionnaires
    - Problèmes de pattern matching multiple
    
Problèmes types:
    - Codeforces: 163E, 590E
    - SPOJ: WPUZZLES
    
Implémentation par: 2025-10-27
Testé: Oui
"""

from collections import deque, defaultdict


class AhoCorasick:
    """
    Algorithme d'Aho-Corasick pour pattern matching multiple.
    """
    
    def __init__(self):
        """Initialise l'automate Aho-Corasick"""
        self.goto = defaultdict(dict)  # goto[state][char] = next_state
        self.fail = {}  # fail[state] = failure_state
        self.output = defaultdict(list)  # output[state] = list of matched patterns
        self.state_count = 0
    
    def add_pattern(self, pattern, pattern_id):
        """
        Ajoute un pattern à l'automate.
        
        Args:
            pattern: Chaîne à ajouter
            pattern_id: Identifiant du pattern
        """
        state = 0
        
        for char in pattern:
            if char not in self.goto[state]:
                self.state_count += 1
                self.goto[state][char] = self.state_count
            state = self.goto[state][char]
        
        self.output[state].append(pattern_id)
    
    def build_failure_links(self):
        """
        Construit les liens de défaillance (failure links).
        Utilise BFS à partir de la racine.
        """
        queue = deque()
        
        # États de profondeur 1 ont une défaillance vers 0
        for char, state in self.goto[0].items():
            self.fail[state] = 0
            queue.append(state)
        
        # BFS pour construire les autres liens
        while queue:
            current_state = queue.popleft()
            
            for char, next_state in self.goto[current_state].items():
                queue.append(next_state)
                
                # Trouver le lien de défaillance
                fail_state = self.fail[current_state]
                
                while fail_state != 0 and char not in self.goto[fail_state]:
                    fail_state = self.fail[fail_state]
                
                if char in self.goto[fail_state]:
                    self.fail[next_state] = self.goto[fail_state][char]
                else:
                    self.fail[next_state] = 0
                
                # Ajouter les outputs du lien de défaillance
                self.output[next_state].extend(self.output[self.fail[next_state]])
    
    def search(self, text):
        """
        Recherche tous les patterns dans le texte.
        
        Args:
            text: Texte dans lequel chercher
            
        Returns:
            Liste de tuples (position, pattern_id) des matchs trouvés
            
        Example:
            >>> ac = AhoCorasick()
            >>> ac.add_pattern("he", 0)
            >>> ac.add_pattern("she", 1)
            >>> ac.add_pattern("his", 2)
            >>> ac.add_pattern("hers", 3)
            >>> ac.build_failure_links()
            >>> matches = ac.search("ushers")
            >>> len(matches) >= 2  # Trouve "she" et "he" et "hers"
            True
        """
        state = 0
        matches = []
        
        for i, char in enumerate(text):
            # Suivre les liens de défaillance si nécessaire
            while state != 0 and char not in self.goto[state]:
                state = self.fail[state]
            
            if char in self.goto[state]:
                state = self.goto[state][char]
            else:
                state = 0
            
            # Enregistrer les matches
            for pattern_id in self.output[state]:
                matches.append((i, pattern_id))
        
        return matches
    
    def count_occurrences(self, text):
        """
        Compte le nombre total d'occurrences de tous les patterns.
        
        Args:
            text: Texte à analyser
            
        Returns:
            Nombre total d'occurrences
        """
        return len(self.search(text))


def find_all_patterns(text, patterns):
    """
    Trouve toutes les occurrences de plusieurs patterns dans un texte.
    
    Args:
        text: Texte dans lequel chercher
        patterns: Liste de patterns
        
    Returns:
        Dict {pattern_id: [(positions)]}
        
    Example:
        >>> text = "ushers"
        >>> patterns = ["he", "she", "his", "hers"]
        >>> result = find_all_patterns(text, patterns)
        >>> 0 in result  # "he" trouvé
        True
    """
    ac = AhoCorasick()
    
    for i, pattern in enumerate(patterns):
        ac.add_pattern(pattern, i)
    
    ac.build_failure_links()
    matches = ac.search(text)
    
    # Organiser par pattern_id
    result = defaultdict(list)
    for pos, pattern_id in matches:
        result[pattern_id].append(pos)
    
    return dict(result)


def filter_bad_words(text, bad_words, replacement='*'):
    """
    Filtre les mots interdits dans un texte.
    
    Args:
        text: Texte original
        bad_words: Liste de mots à censurer
        replacement: Caractère de remplacement
        
    Returns:
        Texte censuré
    """
    ac = AhoCorasick()
    
    # Ajouter tous les mots interdits
    for i, word in enumerate(bad_words):
        ac.add_pattern(word.lower(), i)
    
    ac.build_failure_links()
    
    # Trouver toutes les occurrences
    matches = ac.search(text.lower())
    
    # Marquer les positions à censurer
    to_censor = [False] * len(text)
    for pos, pattern_id in matches:
        word_len = len(bad_words[pattern_id])
        start = pos - word_len + 1
        for i in range(start, pos + 1):
            to_censor[i] = True
    
    # Construire le texte censuré
    result = []
    for i, char in enumerate(text):
        if to_censor[i]:
            result.append(replacement)
        else:
            result.append(char)
    
    return ''.join(result)


def test():
    """Tests unitaires complets"""
    
    # Test basique
    ac = AhoCorasick()
    ac.add_pattern("he", 0)
    ac.add_pattern("she", 1)
    ac.add_pattern("his", 2)
    ac.add_pattern("hers", 3)
    ac.build_failure_links()
    
    matches = ac.search("ushers")
    # Devrait trouver "she" à pos 2, "he" à pos 3, "hers" à pos 5
    assert len(matches) >= 2
    
    # Test count
    count = ac.count_occurrences("ushers")
    assert count >= 2
    
    # Test find_all_patterns
    text = "ahishers"
    patterns = ["he", "she", "his", "hers"]
    result = find_all_patterns(text, patterns)
    
    assert 2 in result  # "his" trouvé
    
    # Test patterns qui se chevauchent
    ac2 = AhoCorasick()
    ac2.add_pattern("abc", 0)
    ac2.add_pattern("bcd", 1)
    ac2.add_pattern("cde", 2)
    ac2.build_failure_links()
    
    matches2 = ac2.search("abcde")
    # Devrait trouver "abc", "bcd", "cde"
    pattern_ids = [m[1] for m in matches2]
    assert 0 in pattern_ids
    assert 1 in pattern_ids
    assert 2 in pattern_ids
    
    # Test filter bad words
    text = "This is a hell of a test"
    bad_words = ["hell"]
    censored = filter_bad_words(text, bad_words)
    assert "****" in censored
    
    # Test avec patterns vides
    ac3 = AhoCorasick()
    ac3.build_failure_links()
    matches3 = ac3.search("test")
    assert len(matches3) == 0
    
    # Test patterns identiques
    ac4 = AhoCorasick()
    ac4.add_pattern("test", 0)
    ac4.add_pattern("test", 1)
    ac4.build_failure_links()
    
    matches4 = ac4.search("test")
    assert len(matches4) == 2  # Deux patterns identiques trouvés
    
    # Test cas sensible
    ac5 = AhoCorasick()
    ac5.add_pattern("Test", 0)
    ac5.build_failure_links()
    
    matches5_lower = ac5.search("test")
    matches5_upper = ac5.search("Test")
    
    # Case sensitive
    assert len(matches5_lower) == 0
    assert len(matches5_upper) >= 1
    
    print("Tous les tests passes")


if __name__ == "__main__":
    test()

