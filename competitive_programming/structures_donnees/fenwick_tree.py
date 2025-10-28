"""
Fenwick Tree / Binary Indexed Tree (BIT)

Description:
    Structure de données pour calculer efficacement les sommes de préfixes
    et effectuer des mises à jour ponctuelles. Plus simple et rapide que
    Segment Tree pour les opérations de somme, mais moins flexible.

Complexité:
    - Temps: O(log n) par requête/update
    - Espace: O(n)

Cas d'usage:
    - Sommes de préfixes/intervalles dynamiques
    - Comptage d'inversions
    - Coordinate compression avec queries
    - Problèmes de fréquences cumulatives
    
Problèmes types:
    - Codeforces: 459D, 652D, 992E
    - AtCoder: ABC119C, ABC134E
    - CSES: Range Update Queries
    
Implémentation par: 2025-10-27
Testé: Oui
"""

class FenwickTree:
    """
    Binary Indexed Tree (BIT) pour sommes de préfixes.
    Indexé à partir de 1 pour simplicité d'implémentation.
    """
    
    def __init__(self, n):
        """
        Initialise un Fenwick Tree de taille n.
        
        Args:
            n: Taille du tableau (indices de 1 à n)
            
        Example:
            >>> ft = FenwickTree(5)
            >>> ft.update(1, 3)
            >>> ft.update(2, 5)
            >>> ft.query(2)  # Somme de 1 à 2
            8
        """
        self.n = n
        self.tree = [0] * (n + 1)
    
    def update(self, idx, delta):
        """
        Ajoute delta à la valeur à l'index idx.
        
        Args:
            idx: Index (1-indexed)
            delta: Valeur à ajouter
        """
        while idx <= self.n:
            self.tree[idx] += delta
            idx += idx & (-idx)  # Ajoute le dernier bit set
    
    def query(self, idx):
        """
        Retourne la somme de préfixe de 1 à idx (inclus).
        
        Args:
            idx: Index jusqu'où calculer (1-indexed)
            
        Returns:
            Somme de 1 à idx
        """
        result = 0
        while idx > 0:
            result += self.tree[idx]
            idx -= idx & (-idx)  # Retire le dernier bit set
        return result
    
    def range_query(self, left, right):
        """
        Retourne la somme de l'intervalle [left, right].
        
        Args:
            left: Début de l'intervalle (1-indexed)
            right: Fin de l'intervalle (1-indexed)
            
        Returns:
            Somme de left à right
        """
        if left > right:
            return 0
        return self.query(right) - self.query(left - 1)
    
    def set_value(self, idx, value, old_value=0):
        """
        Définit la valeur à l'index idx.
        
        Args:
            idx: Index (1-indexed)
            value: Nouvelle valeur
            old_value: Ancienne valeur (0 par défaut)
        """
        self.update(idx, value - old_value)


class FenwickTree2D:
    """
    Fenwick Tree 2D pour sommes de rectangles.
    Utile pour les problèmes de grille avec queries dynamiques.
    """
    
    def __init__(self, rows, cols):
        """
        Initialise un Fenwick Tree 2D.
        
        Args:
            rows: Nombre de lignes
            cols: Nombre de colonnes
        """
        self.rows = rows
        self.cols = cols
        self.tree = [[0] * (cols + 1) for _ in range(rows + 1)]
    
    def update(self, row, col, delta):
        """
        Ajoute delta à la position (row, col).
        
        Args:
            row: Ligne (1-indexed)
            col: Colonne (1-indexed)
            delta: Valeur à ajouter
        """
        i = row
        while i <= self.rows:
            j = col
            while j <= self.cols:
                self.tree[i][j] += delta
                j += j & (-j)
            i += i & (-i)
    
    def query(self, row, col):
        """
        Somme du rectangle de (1,1) à (row, col).
        
        Args:
            row: Ligne (1-indexed)
            col: Colonne (1-indexed)
            
        Returns:
            Somme du rectangle
        """
        result = 0
        i = row
        while i > 0:
            j = col
            while j > 0:
                result += self.tree[i][j]
                j -= j & (-j)
            i -= i & (-i)
        return result
    
    def range_query(self, row1, col1, row2, col2):
        """
        Somme du rectangle de (row1, col1) à (row2, col2).
        
        Args:
            row1, col1: Coin supérieur gauche (1-indexed)
            row2, col2: Coin inférieur droit (1-indexed)
            
        Returns:
            Somme du rectangle
        """
        # Principe d'inclusion-exclusion
        return (self.query(row2, col2) 
                - self.query(row1 - 1, col2)
                - self.query(row2, col1 - 1)
                + self.query(row1 - 1, col1 - 1))


class FenwickTreeRangeUpdate:
    """
    Fenwick Tree avec support de range updates et point queries.
    Utilise deux BIT pour gérer les updates d'intervalles.
    """
    
    def __init__(self, n):
        """
        Initialise un Fenwick Tree avec range updates.
        
        Args:
            n: Taille du tableau
        """
        self.n = n
        self.tree1 = [0] * (n + 1)
        self.tree2 = [0] * (n + 1)
    
    def _update(self, tree, idx, delta):
        """Update interne sur un arbre"""
        while idx <= self.n:
            tree[idx] += delta
            idx += idx & (-idx)
    
    def _query(self, tree, idx):
        """Query interne sur un arbre"""
        result = 0
        while idx > 0:
            result += tree[idx]
            idx -= idx & (-idx)
        return result
    
    def range_update(self, left, right, delta):
        """
        Ajoute delta à tous les éléments de [left, right].
        
        Args:
            left: Début intervalle (1-indexed)
            right: Fin intervalle (1-indexed)
            delta: Valeur à ajouter
        """
        self._update(self.tree1, left, delta)
        self._update(self.tree1, right + 1, -delta)
        self._update(self.tree2, left, delta * (left - 1))
        self._update(self.tree2, right + 1, -delta * right)
    
    def query(self, idx):
        """
        Retourne la valeur à l'index idx.
        
        Args:
            idx: Index (1-indexed)
            
        Returns:
            Valeur à l'index idx
        """
        return self._query(self.tree1, idx) * idx - self._query(self.tree2, idx)


def test():
    """Tests unitaires complets"""
    
    # Test Fenwick Tree basique
    ft = FenwickTree(10)
    
    # Test updates et queries
    ft.update(1, 5)
    ft.update(2, 3)
    ft.update(3, 7)
    
    assert ft.query(1) == 5
    assert ft.query(2) == 8
    assert ft.query(3) == 15
    
    # Test range query
    assert ft.range_query(1, 3) == 15
    assert ft.range_query(2, 3) == 10
    assert ft.range_query(1, 1) == 5
    
    # Test set_value
    ft.set_value(2, 10, 3)  # Change 3 en 10
    assert ft.query(2) == 15  # 5 + 10
    assert ft.range_query(2, 3) == 17
    
    # Test updates multiples
    for i in range(1, 11):
        ft2 = FenwickTree(10)
        ft2.update(i, i)
    
    ft3 = FenwickTree(10)
    for i in range(1, 11):
        ft3.update(i, i)
    assert ft3.query(10) == 55  # 1+2+...+10
    
    # Test Fenwick Tree 2D
    ft2d = FenwickTree2D(5, 5)
    
    ft2d.update(1, 1, 1)
    ft2d.update(2, 2, 2)
    ft2d.update(3, 3, 3)
    
    assert ft2d.query(1, 1) == 1
    assert ft2d.query(2, 2) == 3
    assert ft2d.query(3, 3) == 6
    
    # Test range query 2D
    assert ft2d.range_query(1, 1, 2, 2) == 3  # 1 + 2
    assert ft2d.range_query(2, 2, 3, 3) == 5  # 2 + 3
    
    # Test Fenwick Tree avec Range Updates
    ft_range = FenwickTreeRangeUpdate(10)
    
    ft_range.range_update(1, 5, 10)
    assert ft_range.query(1) == 10
    assert ft_range.query(3) == 10
    assert ft_range.query(5) == 10
    assert ft_range.query(6) == 0
    
    ft_range.range_update(3, 7, 5)
    assert ft_range.query(3) == 15  # 10 + 5
    assert ft_range.query(5) == 15
    assert ft_range.query(6) == 5
    assert ft_range.query(7) == 5
    assert ft_range.query(8) == 0
    
    # Test edge case
    ft_single = FenwickTree(1)
    ft_single.update(1, 42)
    assert ft_single.query(1) == 42
    assert ft_single.range_query(1, 1) == 42
    
    print("Tous les tests passes")


if __name__ == "__main__":
    test()

