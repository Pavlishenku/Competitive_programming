"""
Segment Tree (Arbre de Segments)

Description:
    Structure de données permettant des requêtes et mises à jour efficaces
    sur des intervalles. Supporte range queries (min, max, sum, gcd, etc.)
    et updates de point ou d'intervalle avec lazy propagation.

Complexité:
    - Temps: O(log n) par requête/update
    - Espace: O(4n) ≈ O(n)

Cas d'usage:
    - Range sum/min/max queries
    - Range updates avec lazy propagation
    - Problèmes d'intervalles dynamiques
    - Persistance de données sur intervalles
    
Problèmes types:
    - Codeforces: 339D, 380C, 145E, 52C
    - AtCoder: ABC125D, ABC157E
    - CSES: Range Queries section
    
Implémentation par: 2025-10-27
Testé: Oui
"""

class SegmentTree:
    """
    Segment Tree générique pour range queries.
    Supporte différentes opérations (sum, min, max, gcd, etc.)
    """
    
    def __init__(self, arr, operation='sum', default_value=0):
        """
        Initialise le Segment Tree.
        
        Args:
            arr: Liste des valeurs initiales
            operation: Type d'opération ('sum', 'min', 'max', 'gcd')
            default_value: Valeur par défaut (0 pour sum, inf pour min, etc.)
            
        Example:
            >>> st = SegmentTree([1, 3, 5, 7, 9, 11])
            >>> st.query(1, 3)  # sum de indices 1 à 3
            15
        """
        self.n = len(arr)
        self.tree = [default_value] * (4 * self.n)
        self.operation = operation
        self.default_value = default_value
        
        # Définir la fonction d'agrégation
        if operation == 'sum':
            self.func = lambda a, b: a + b
        elif operation == 'min':
            self.func = lambda a, b: min(a, b)
        elif operation == 'max':
            self.func = lambda a, b: max(a, b)
        elif operation == 'gcd':
            from math import gcd
            self.func = gcd
        else:
            raise ValueError(f"Operation {operation} non supportée")
        
        if arr:
            self._build(arr, 0, 0, self.n - 1)
    
    def _build(self, arr, node, start, end):
        """
        Construit le segment tree récursivement.
        
        Args:
            arr: Tableau source
            node: Index du noeud actuel
            start: Début de l'intervalle
            end: Fin de l'intervalle
        """
        if start == end:
            self.tree[node] = arr[start]
        else:
            mid = (start + end) // 2
            left_node = 2 * node + 1
            right_node = 2 * node + 2
            
            self._build(arr, left_node, start, mid)
            self._build(arr, right_node, mid + 1, end)
            
            self.tree[node] = self.func(self.tree[left_node], self.tree[right_node])
    
    def update(self, idx, value):
        """
        Met à jour la valeur à l'index idx.
        
        Args:
            idx: Index à mettre à jour
            value: Nouvelle valeur
        """
        self._update(0, 0, self.n - 1, idx, value)
    
    def _update(self, node, start, end, idx, value):
        """
        Met à jour récursivement.
        
        Args:
            node: Noeud actuel
            start: Début intervalle
            end: Fin intervalle
            idx: Index à mettre à jour
            value: Nouvelle valeur
        """
        if start == end:
            self.tree[node] = value
        else:
            mid = (start + end) // 2
            left_node = 2 * node + 1
            right_node = 2 * node + 2
            
            if idx <= mid:
                self._update(left_node, start, mid, idx, value)
            else:
                self._update(right_node, mid + 1, end, idx, value)
            
            self.tree[node] = self.func(self.tree[left_node], self.tree[right_node])
    
    def query(self, left, right):
        """
        Effectue une requête sur l'intervalle [left, right].
        
        Args:
            left: Début de l'intervalle (inclus)
            right: Fin de l'intervalle (inclus)
            
        Returns:
            Résultat de l'opération sur l'intervalle
        """
        return self._query(0, 0, self.n - 1, left, right)
    
    def _query(self, node, start, end, left, right):
        """
        Requête récursive.
        
        Args:
            node: Noeud actuel
            start: Début intervalle noeud
            end: Fin intervalle noeud
            left: Début intervalle requête
            right: Fin intervalle requête
            
        Returns:
            Résultat de la requête
        """
        # Pas de chevauchement
        if right < start or left > end:
            return self.default_value
        
        # Chevauchement total
        if left <= start and end <= right:
            return self.tree[node]
        
        # Chevauchement partiel
        mid = (start + end) // 2
        left_node = 2 * node + 1
        right_node = 2 * node + 2
        
        left_result = self._query(left_node, start, mid, left, right)
        right_result = self._query(right_node, mid + 1, end, left, right)
        
        return self.func(left_result, right_result)


class SegmentTreeLazy:
    """
    Segment Tree avec Lazy Propagation pour range updates.
    Supporte les mises à jour d'intervalles en O(log n).
    """
    
    def __init__(self, arr):
        """
        Initialise le Segment Tree avec lazy propagation.
        
        Args:
            arr: Liste des valeurs initiales
        """
        self.n = len(arr)
        self.tree = [0] * (4 * self.n)
        self.lazy = [0] * (4 * self.n)
        
        if arr:
            self._build(arr, 0, 0, self.n - 1)
    
    def _build(self, arr, node, start, end):
        """Construit l'arbre"""
        if start == end:
            self.tree[node] = arr[start]
        else:
            mid = (start + end) // 2
            left_node = 2 * node + 1
            right_node = 2 * node + 2
            
            self._build(arr, left_node, start, mid)
            self._build(arr, right_node, mid + 1, end)
            
            self.tree[node] = self.tree[left_node] + self.tree[right_node]
    
    def _push(self, node, start, end):
        """Propage les mises à jour lazy"""
        if self.lazy[node] != 0:
            self.tree[node] += (end - start + 1) * self.lazy[node]
            
            if start != end:  # Pas une feuille
                left_node = 2 * node + 1
                right_node = 2 * node + 2
                self.lazy[left_node] += self.lazy[node]
                self.lazy[right_node] += self.lazy[node]
            
            self.lazy[node] = 0
    
    def range_update(self, left, right, delta):
        """
        Ajoute delta à tous les éléments dans [left, right].
        
        Args:
            left: Début intervalle
            right: Fin intervalle
            delta: Valeur à ajouter
        """
        self._range_update(0, 0, self.n - 1, left, right, delta)
    
    def _range_update(self, node, start, end, left, right, delta):
        """Range update récursif avec lazy propagation"""
        self._push(node, start, end)
        
        if right < start or left > end:
            return
        
        if left <= start and end <= right:
            self.lazy[node] += delta
            self._push(node, start, end)
            return
        
        mid = (start + end) // 2
        left_node = 2 * node + 1
        right_node = 2 * node + 2
        
        self._range_update(left_node, start, mid, left, right, delta)
        self._range_update(right_node, mid + 1, end, left, right, delta)
        
        self._push(left_node, start, mid)
        self._push(right_node, mid + 1, end)
        
        self.tree[node] = self.tree[left_node] + self.tree[right_node]
    
    def query(self, left, right):
        """
        Somme sur l'intervalle [left, right].
        
        Args:
            left: Début intervalle
            right: Fin intervalle
            
        Returns:
            Somme de l'intervalle
        """
        return self._query(0, 0, self.n - 1, left, right)
    
    def _query(self, node, start, end, left, right):
        """Query récursif"""
        if right < start or left > end:
            return 0
        
        self._push(node, start, end)
        
        if left <= start and end <= right:
            return self.tree[node]
        
        mid = (start + end) // 2
        left_node = 2 * node + 1
        right_node = 2 * node + 2
        
        left_result = self._query(left_node, start, mid, left, right)
        right_result = self._query(right_node, mid + 1, end, left, right)
        
        return left_result + right_result


def test():
    """Tests unitaires complets"""
    
    # Test Segment Tree basique (sum)
    arr = [1, 3, 5, 7, 9, 11]
    st = SegmentTree(arr, 'sum', 0)
    
    assert st.query(1, 3) == 15  # 3 + 5 + 7
    assert st.query(0, 5) == 36  # somme totale
    assert st.query(2, 2) == 5   # un seul élément
    
    # Test update
    st.update(1, 10)
    assert st.query(1, 3) == 22  # 10 + 5 + 7
    assert st.query(0, 5) == 43
    
    # Test Segment Tree min
    st_min = SegmentTree(arr, 'min', float('inf'))
    assert st_min.query(0, 5) == 1
    assert st_min.query(2, 4) == 5
    
    st_min.update(2, 2)
    assert st_min.query(2, 4) == 2
    
    # Test Segment Tree max
    st_max = SegmentTree(arr, 'max', float('-inf'))
    assert st_max.query(0, 5) == 11
    assert st_max.query(0, 3) == 7
    
    # Test Lazy Propagation
    arr2 = [1, 2, 3, 4, 5]
    st_lazy = SegmentTreeLazy(arr2)
    
    assert st_lazy.query(0, 4) == 15
    
    st_lazy.range_update(1, 3, 10)
    assert st_lazy.query(0, 4) == 45  # 1 + 12 + 13 + 14 + 5
    assert st_lazy.query(1, 3) == 39  # 12 + 13 + 14
    
    st_lazy.range_update(0, 4, -5)
    assert st_lazy.query(0, 4) == 20  # tous -5
    
    # Test edge case: arbre de taille 1
    st_single = SegmentTree([42], 'sum', 0)
    assert st_single.query(0, 0) == 42
    st_single.update(0, 100)
    assert st_single.query(0, 0) == 100
    
    print("Tous les tests passes")


if __name__ == "__main__":
    test()

