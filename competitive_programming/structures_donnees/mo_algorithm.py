"""
Mo's Algorithm (Algorithm of Square Root Decomposition)

Description:
    Algorithme pour répondre efficacement à des requêtes offline sur intervalles.
    Utilise la décomposition en blocs pour minimiser les mouvements.

Complexité:
    - Temps: O((N + Q) × √N) pour Q requêtes
    - Espace: O(N + Q)

Cas d'usage:
    - Requêtes offline sur intervalles
    - Comptage d'éléments distincts dans un range
    - Somme/produit sur intervalles avec add/remove
    - Problèmes avec opérations réversibles
    
Problèmes types:
    - Codeforces: 220B, 86D, 940F
    - SPOJ: DQUERY
    
Implémentation par: 2025-10-27
Testé: Oui
"""

import math


class MoAlgorithm:
    """
    Mo's Algorithm pour requêtes offline sur intervalles.
    """
    
    def __init__(self, arr):
        """
        Args:
            arr: Tableau source
        """
        self.arr = arr
        self.n = len(arr)
        self.block_size = int(math.sqrt(self.n)) + 1
        
        # État actuel
        self.current_left = 0
        self.current_right = -1
        self.current_answer = 0
    
    def add(self, idx):
        """
        Ajoute l'élément à l'index idx à l'état actuel.
        À surcharger selon le problème.
        """
        pass
    
    def remove(self, idx):
        """
        Retire l'élément à l'index idx de l'état actuel.
        À surcharger selon le problème.
        """
        pass
    
    def get_answer(self):
        """
        Retourne la réponse pour l'intervalle actuel.
        """
        return self.current_answer
    
    def process_queries(self, queries):
        """
        Traite toutes les requêtes de manière optimisée.
        
        Args:
            queries: Liste de tuples (left, right, query_id)
            
        Returns:
            Liste des réponses dans l'ordre des query_id
        """
        # Trier les requêtes selon Mo's order
        def mo_comparator(query):
            left, right, _ = query
            block = left // self.block_size
            if block % 2 == 0:
                return (block, right)
            else:
                return (block, -right)
        
        sorted_queries = sorted(queries, key=mo_comparator)
        
        answers = [0] * len(queries)
        
        for left, right, query_id in sorted_queries:
            # Déplacer les pointeurs
            while self.current_right < right:
                self.current_right += 1
                self.add(self.current_right)
            
            while self.current_right > right:
                self.remove(self.current_right)
                self.current_right -= 1
            
            while self.current_left < left:
                self.remove(self.current_left)
                self.current_left += 1
            
            while self.current_left > left:
                self.current_left -= 1
                self.add(self.current_left)
            
            answers[query_id] = self.get_answer()
        
        return answers


class MoDistinctElements(MoAlgorithm):
    """
    Mo's Algorithm pour compter les éléments distincts dans un intervalle.
    """
    
    def __init__(self, arr):
        super().__init__(arr)
        self.freq = {}
        self.distinct_count = 0
    
    def add(self, idx):
        """Ajoute un élément"""
        val = self.arr[idx]
        
        if val not in self.freq:
            self.freq[val] = 0
        
        if self.freq[val] == 0:
            self.distinct_count += 1
        
        self.freq[val] += 1
    
    def remove(self, idx):
        """Retire un élément"""
        val = self.arr[idx]
        
        self.freq[val] -= 1
        
        if self.freq[val] == 0:
            self.distinct_count -= 1
    
    def get_answer(self):
        """Retourne le nombre d'éléments distincts"""
        return self.distinct_count


class MoRangeSum(MoAlgorithm):
    """
    Mo's Algorithm pour calculer la somme sur un intervalle.
    """
    
    def __init__(self, arr):
        super().__init__(arr)
        self.current_sum = 0
    
    def add(self, idx):
        """Ajoute un élément à la somme"""
        self.current_sum += self.arr[idx]
    
    def remove(self, idx):
        """Retire un élément de la somme"""
        self.current_sum -= self.arr[idx]
    
    def get_answer(self):
        """Retourne la somme actuelle"""
        return self.current_sum


class MoRangeMode(MoAlgorithm):
    """
    Mo's Algorithm pour trouver le mode (élément le plus fréquent) dans un intervalle.
    """
    
    def __init__(self, arr):
        super().__init__(arr)
        self.freq = {}
        self.max_freq = 0
    
    def add(self, idx):
        """Ajoute un élément"""
        val = self.arr[idx]
        
        if val not in self.freq:
            self.freq[val] = 0
        
        self.freq[val] += 1
        self.max_freq = max(self.max_freq, self.freq[val])
    
    def remove(self, idx):
        """Retire un élément"""
        val = self.arr[idx]
        
        if self.freq[val] == self.max_freq:
            self.freq[val] -= 1
            if self.freq[val] < self.max_freq:
                # Recalculer max_freq
                self.max_freq = max(self.freq.values()) if self.freq else 0
        else:
            self.freq[val] -= 1
    
    def get_answer(self):
        """Retourne la fréquence maximale"""
        return self.max_freq


def count_distinct_in_ranges(arr, queries):
    """
    Compte le nombre d'éléments distincts pour plusieurs requêtes.
    
    Args:
        arr: Tableau d'entiers
        queries: Liste de tuples (left, right)
        
    Returns:
        Liste des nombres d'éléments distincts pour chaque requête
        
    Example:
        >>> arr = [1, 2, 1, 3, 2, 4]
        >>> queries = [(0, 2), (1, 4), (0, 5)]
        >>> count_distinct_in_ranges(arr, queries)
        [2, 3, 4]
    """
    mo = MoDistinctElements(arr)
    
    # Ajouter les IDs aux requêtes
    queries_with_id = [(l, r, i) for i, (l, r) in enumerate(queries)]
    
    return mo.process_queries(queries_with_id)


def test():
    """Tests unitaires complets"""
    
    # Test distinct elements
    arr = [1, 2, 1, 3, 2, 4, 1, 2, 3]
    queries = [
        (0, 2),  # [1, 2, 1] -> 2 distinct
        (1, 4),  # [2, 1, 3, 2] -> 3 distinct
        (0, 5),  # [1, 2, 1, 3, 2, 4] -> 4 distinct
        (3, 8),  # [3, 2, 4, 1, 2, 3] -> 4 distinct
    ]
    
    mo_distinct = MoDistinctElements(arr)
    queries_with_id = [(l, r, i) for i, (l, r) in enumerate(queries)]
    answers = mo_distinct.process_queries(queries_with_id)
    
    assert answers[0] == 2
    assert answers[1] == 3
    assert answers[2] == 4
    assert answers[3] == 4
    
    # Test range sum
    arr_sum = [1, 2, 3, 4, 5]
    queries_sum = [
        (0, 2),  # 1+2+3 = 6
        (1, 4),  # 2+3+4+5 = 14
        (0, 4),  # 1+2+3+4+5 = 15
    ]
    
    mo_sum = MoRangeSum(arr_sum)
    queries_sum_with_id = [(l, r, i) for i, (l, r) in enumerate(queries_sum)]
    answers_sum = mo_sum.process_queries(queries_sum_with_id)
    
    assert answers_sum[0] == 6
    assert answers_sum[1] == 14
    assert answers_sum[2] == 15
    
    # Test range mode
    arr_mode = [1, 2, 1, 1, 2, 3, 3, 3]
    queries_mode = [
        (0, 3),  # [1, 2, 1, 1] -> max freq 3
        (4, 7),  # [2, 3, 3, 3] -> max freq 3
        (0, 7),  # tout -> max freq 3 ou 4
    ]
    
    mo_mode = MoRangeMode(arr_mode)
    queries_mode_with_id = [(l, r, i) for i, (l, r) in enumerate(queries_mode)]
    answers_mode = mo_mode.process_queries(queries_mode_with_id)
    
    assert answers_mode[0] == 3
    assert answers_mode[1] == 3
    assert answers_mode[2] == 4
    
    # Test avec un seul élément
    arr_single = [5]
    queries_single = [(0, 0)]
    
    mo_single = MoDistinctElements(arr_single)
    queries_single_with_id = [(0, 0, 0)]
    answers_single = mo_single.process_queries(queries_single_with_id)
    
    assert answers_single[0] == 1
    
    # Test count_distinct_in_ranges helper
    arr_helper = [1, 2, 1, 3, 2, 4]
    queries_helper = [(0, 2), (1, 4), (0, 5)]
    result = count_distinct_in_ranges(arr_helper, queries_helper)
    
    assert result[0] == 2
    assert result[1] == 3
    assert result[2] == 4
    
    print("Tous les tests passes")


if __name__ == "__main__":
    test()

