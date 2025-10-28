"""
Trie (Arbre de Préfixes)

Description:
    Structure de données pour stocker et rechercher efficacement des chaînes.
    Permet recherche de préfixes, autocomplétion, dictionnaires.

Complexité:
    - Insertion/Recherche: O(L) où L = longueur de la chaîne
    - Espace: O(N*L*A) où A = taille de l'alphabet

Cas d'usage:
    - Dictionnaires avec recherche de préfixes
    - Autocomplétion
    - Comptage de mots avec préfixe commun
    - XOR maximum (Trie binaire)
    
Problèmes types:
    - Codeforces: 706D, 948D, 1285D
    - AtCoder: ABC287E
    - CSES: String Matching
    
Implémentation par: 2025-10-27
Testé: Oui
"""


class TrieNode:
    """Noeud d'un Trie"""
    
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False
        self.count = 0  # Nombre de mots passant par ce noeud


class Trie:
    """
    Trie pour recherche de chaînes et préfixes.
    """
    
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, word):
        """
        Insère un mot dans le Trie.
        
        Args:
            word: Mot à insérer
            
        Example:
            >>> trie = Trie()
            >>> trie.insert("hello")
            >>> trie.search("hello")
            True
        """
        node = self.root
        
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
            node.count += 1
        
        node.is_end_of_word = True
    
    def search(self, word):
        """
        Recherche un mot exact dans le Trie.
        
        Args:
            word: Mot à rechercher
            
        Returns:
            True si le mot existe, False sinon
        """
        node = self.root
        
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        
        return node.is_end_of_word
    
    def starts_with(self, prefix):
        """
        Vérifie si un préfixe existe.
        
        Args:
            prefix: Préfixe à chercher
            
        Returns:
            True si le préfixe existe
        """
        node = self.root
        
        for char in prefix:
            if char not in node.children:
                return False
            node = node.children[char]
        
        return True
    
    def count_words_with_prefix(self, prefix):
        """
        Compte le nombre de mots avec un préfixe donné.
        
        Args:
            prefix: Préfixe
            
        Returns:
            Nombre de mots avec ce préfixe
        """
        node = self.root
        
        for char in prefix:
            if char not in node.children:
                return 0
            node = node.children[char]
        
        return node.count
    
    def delete(self, word):
        """
        Supprime un mot du Trie.
        
        Args:
            word: Mot à supprimer
            
        Returns:
            True si supprimé, False si n'existait pas
        """
        def _delete(node, word, depth):
            if depth == len(word):
                if not node.is_end_of_word:
                    return False
                node.is_end_of_word = False
                return len(node.children) == 0
            
            char = word[depth]
            if char not in node.children:
                return False
            
            child = node.children[char]
            should_delete = _delete(child, word, depth + 1)
            
            if should_delete:
                del node.children[char]
                return len(node.children) == 0 and not node.is_end_of_word
            
            return False
        
        return _delete(self.root, word, 0)
    
    def get_all_words(self):
        """
        Retourne tous les mots du Trie.
        
        Returns:
            Liste de tous les mots
        """
        words = []
        
        def dfs(node, current_word):
            if node.is_end_of_word:
                words.append(current_word)
            
            for char, child in node.children.items():
                dfs(child, current_word + char)
        
        dfs(self.root, "")
        return words


class BinaryTrie:
    """
    Trie binaire pour opérations XOR.
    Utilisé pour trouver XOR maximum dans un ensemble.
    """
    
    def __init__(self, max_bits=31):
        """
        Args:
            max_bits: Nombre de bits maximum (31 pour int 32-bit)
        """
        self.root = TrieNode()
        self.max_bits = max_bits
    
    def insert(self, num):
        """
        Insère un nombre dans le Trie binaire.
        
        Args:
            num: Nombre à insérer
        """
        node = self.root
        
        for i in range(self.max_bits, -1, -1):
            bit = (num >> i) & 1
            bit_str = str(bit)
            
            if bit_str not in node.children:
                node.children[bit_str] = TrieNode()
            
            node = node.children[bit_str]
    
    def find_max_xor(self, num):
        """
        Trouve le XOR maximum de num avec un nombre dans le Trie.
        
        Args:
            num: Nombre de référence
            
        Returns:
            XOR maximum possible
        """
        node = self.root
        xor_value = 0
        
        for i in range(self.max_bits, -1, -1):
            bit = (num >> i) & 1
            # Chercher le bit opposé pour maximiser XOR
            opposite_bit = str(1 - bit)
            
            if opposite_bit in node.children:
                xor_value |= (1 << i)
                node = node.children[opposite_bit]
            elif str(bit) in node.children:
                node = node.children[str(bit)]
            else:
                # Trie vide
                return 0
        
        return xor_value
    
    def find_max_xor_pair(self, nums):
        """
        Trouve la paire avec XOR maximum dans une liste.
        
        Args:
            nums: Liste de nombres
            
        Returns:
            XOR maximum entre deux nombres
        """
        if len(nums) < 2:
            return 0
        
        max_xor = 0
        
        for num in nums:
            self.insert(num)
            max_xor = max(max_xor, self.find_max_xor(num))
        
        return max_xor


def longest_common_prefix(words):
    """
    Trouve le plus long préfixe commun en utilisant un Trie.
    
    Args:
        words: Liste de mots
        
    Returns:
        Plus long préfixe commun
    """
    if not words:
        return ""
    
    trie = Trie()
    for word in words:
        trie.insert(word)
    
    # Parcourir le Trie jusqu'à une branche
    node = trie.root
    prefix = []
    
    while len(node.children) == 1 and not node.is_end_of_word:
        char = list(node.children.keys())[0]
        prefix.append(char)
        node = node.children[char]
    
    return ''.join(prefix)


def test():
    """Tests unitaires complets"""
    
    # Test Trie basique
    trie = Trie()
    
    trie.insert("apple")
    trie.insert("app")
    trie.insert("apricot")
    trie.insert("banana")
    
    assert trie.search("apple")
    assert trie.search("app")
    assert not trie.search("appl")
    assert trie.search("banana")
    
    # Test starts_with
    assert trie.starts_with("app")
    assert trie.starts_with("ban")
    assert not trie.starts_with("ora")
    
    # Test count
    assert trie.count_words_with_prefix("app") == 2
    assert trie.count_words_with_prefix("ban") == 1
    
    # Test get_all_words
    words = sorted(trie.get_all_words())
    assert words == ["app", "apple", "apricot", "banana"]
    
    # Test delete
    trie.delete("app")
    assert not trie.search("app")
    assert trie.search("apple")
    
    # Test Binary Trie
    btrie = BinaryTrie(max_bits=5)
    
    # XOR maximum
    nums = [3, 10, 5, 25, 2, 8]
    max_xor = btrie.find_max_xor_pair(nums)
    assert max_xor == 28  # 5 XOR 25 = 28
    
    # Test single number
    btrie2 = BinaryTrie()
    btrie2.insert(5)
    assert btrie2.find_max_xor(3) == 6  # 5 XOR 3 = 6
    
    # Test longest common prefix
    words1 = ["flower", "flow", "flight"]
    assert longest_common_prefix(words1) == "fl"
    
    words2 = ["dog", "racecar", "car"]
    assert longest_common_prefix(words2) == ""
    
    # Test edge cases
    trie_empty = Trie()
    assert not trie_empty.search("test")
    assert trie_empty.get_all_words() == []
    
    print("Tous les tests passes")


if __name__ == "__main__":
    test()

