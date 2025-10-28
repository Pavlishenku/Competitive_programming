"""
Sparse Table (Table Creuse)

Description:
    Structure de données pour RMQ (Range Minimum Query) statiques en O(1).
    Précalcule les minimums pour tous les intervalles de longueur 2^k.
    Idempotent: min(a, a) = a, donc on peut chevaucher les intervalles.

Complexité:
    - Preprocessing: O(n log n)
    - Query: O(1)
    - Espace: O(n log n)

Cas d'usage:
    - RMQ statiques (pas d'updates)
    - Range GCD/LCM queries
    - LCA avec Euler tour
    
Problèmes types:
    - Codeforces: 872B, 1187D
    - CSES: Static Range Minimum Queries
    
Implémentation par: 2025-10-27
Testé: Oui
"""

import math


class SparseTable:
    """
    Sparse Table pour RMQ en O(1).
    Supporte différentes opérations (min, max, gcd).
    """
    
    def __init__(self, arr, operation='min'):
        """
        Args:
            arr: Tableau source
            operation: 'min', 'max', ou 'gcd'
            
        Example:
            >>> st = SparseTable([1, 3, 2, 7, 9, 11])
            >>> st.query(1, 4)
            2
        """
        self.n = len(arr)
        self.operation = operation
        
        if self.n == 0:
            self.log = []
            self.table = []
            return
        
        # Précalculer les log
        self.log = [0] * (self.n + 1)
        for i in range(2, self.n + 1):
            self.log[i] = self.log[i // 2] + 1
        
        # Nombre de niveaux
        max_log = self.log[self.n] + 1
        
        # table[i][j] = résultat sur [j, j + 2^i)
        self.table = [[0] * self.n for _ in range(max_log)]
        
        # Définir la fonction
        if operation == 'min':
            self.func = min
        elif operation == 'max':
            self.func = max
        elif operation == 'gcd':
            from math import gcd
            self.func = gcd
        else:
            raise ValueError(f"Operation {operation} non supportée")
        
        # Initialiser le premier niveau
        for i in range(self.n):
            self.table[0][i] = arr[i]
        
        # Remplir la table
        for i in range(1, max_log):
            j = 0
            while j + (1 << i) <= self.n:
                self.table[i][j] = self.func(
                    self.table[i-1][j],
                    self.table[i-1][j + (1 << (i-1))]
                )
                j += 1
    
    def query(self, left, right):
        """
        Query sur l'intervalle [left, right] (inclusif).
        
        Args:
            left: Début de l'intervalle
            right: Fin de l'intervalle (inclus)
            
        Returns:
            Résultat de l'opération sur [left, right]
        """
        if left > right or right >= self.n:
            raise ValueError("Intervalle invalide")
        
        # Longueur de l'intervalle
        length = right - left + 1
        k = self.log[length]
        
        # Combiner deux intervalles qui se chevauchent
        return self.func(
            self.table[k][left],
            self.table[k][right - (1 << k) + 1]
        )


class SparseTable2D:
    """
    Sparse Table 2D pour RMQ sur matrices.
    """
    
    def __init__(self, matrix, operation='min'):
        """
        Args:
            matrix: Matrice 2D
            operation: 'min' ou 'max'
        """
        if not matrix or not matrix[0]:
            self.rows = 0
            self.cols = 0
            return
        
        self.rows = len(matrix)
        self.cols = len(matrix[0])
        self.operation = operation
        
        # Précalculer les logs
        self.log_row = [0] * (self.rows + 1)
        self.log_col = [0] * (self.cols + 1)
        
        for i in range(2, self.rows + 1):
            self.log_row[i] = self.log_row[i // 2] + 1
        for i in range(2, self.cols + 1):
            self.log_col[i] = self.log_col[i // 2] + 1
        
        max_log_row = self.log_row[self.rows] + 1
        max_log_col = self.log_col[self.cols] + 1
        
        # table[k1][k2][i][j] = résultat sur rectangle
        # de (i, j) avec taille (2^k1, 2^k2)
        self.table = [[[[0] * self.cols for _ in range(self.rows)]
                      for _ in range(max_log_col)]
                     for _ in range(max_log_row)]
        
        # Fonction
        self.func = min if operation == 'min' else max
        
        # Initialiser
        for i in range(self.rows):
            for j in range(self.cols):
                self.table[0][0][i][j] = matrix[i][j]
        
        # Remplir selon les colonnes
        for k in range(1, max_log_col):
            for i in range(self.rows):
                j = 0
                while j + (1 << k) <= self.cols:
                    self.table[0][k][i][j] = self.func(
                        self.table[0][k-1][i][j],
                        self.table[0][k-1][i][j + (1 << (k-1))]
                    )
                    j += 1
        
        # Remplir selon les lignes
        for k1 in range(1, max_log_row):
            for k2 in range(max_log_col):
                i = 0
                while i + (1 << k1) <= self.rows:
                    j = 0
                    while j + (1 << k2) <= self.cols:
                        self.table[k1][k2][i][j] = self.func(
                            self.table[k1-1][k2][i][j],
                            self.table[k1-1][k2][i + (1 << (k1-1))][j]
                        )
                        j += 1
                    i += 1
    
    def query(self, r1, c1, r2, c2):
        """
        Query sur le rectangle (r1, c1) à (r2, c2) (inclusif).
        
        Args:
            r1, c1: Coin supérieur gauche
            r2, c2: Coin inférieur droit
            
        Returns:
            Résultat de l'opération
        """
        k1 = self.log_row[r2 - r1 + 1]
        k2 = self.log_col[c2 - c1 + 1]
        
        # Combiner 4 rectangles qui se chevauchent
        return self.func(
            self.func(
                self.table[k1][k2][r1][c1],
                self.table[k1][k2][r1][c2 - (1 << k2) + 1]
            ),
            self.func(
                self.table[k1][k2][r2 - (1 << k1) + 1][c1],
                self.table[k1][k2][r2 - (1 << k1) + 1][c2 - (1 << k2) + 1]
            )
        )


def range_gcd(arr):
    """
    Crée une Sparse Table pour range GCD queries.
    
    Args:
        arr: Tableau d'entiers
        
    Returns:
        SparseTable pour GCD
    """
    return SparseTable(arr, operation='gcd')


def test():
    """Tests unitaires complets"""
    
    # Test Sparse Table min
    arr = [1, 3, 2, 7, 9, 11, 3, 5]
    st_min = SparseTable(arr, 'min')
    
    assert st_min.query(0, 2) == 1
    assert st_min.query(1, 4) == 2
    assert st_min.query(3, 7) == 3
    assert st_min.query(0, 7) == 1
    
    # Test Sparse Table max
    st_max = SparseTable(arr, 'max')
    
    assert st_max.query(0, 2) == 3
    assert st_max.query(3, 5) == 11
    assert st_max.query(0, 7) == 11
    
    # Test Sparse Table GCD
    arr_gcd = [12, 18, 24, 30, 36]
    st_gcd = SparseTable(arr_gcd, 'gcd')
    
    assert st_gcd.query(0, 2) == 6
    assert st_gcd.query(1, 3) == 6
    assert st_gcd.query(0, 4) == 6
    
    # Test puissance de 2
    arr2 = [1, 2, 3, 4]
    st2 = SparseTable(arr2, 'min')
    assert st2.query(0, 3) == 1
    assert st2.query(2, 3) == 3
    
    # Test Sparse Table 2D
    matrix = [
        [1, 2, 3, 4],
        [5, 0, 7, 8],
        [9, 10, 2, 12],
        [13, 14, 15, 1]
    ]
    
    st2d = SparseTable2D(matrix, 'min')
    
    assert st2d.query(0, 0, 1, 1) == 0
    assert st2d.query(0, 0, 3, 3) == 0
    assert st2d.query(2, 2, 3, 3) == 1
    
    # Test 2D max
    st2d_max = SparseTable2D(matrix, 'max')
    assert st2d_max.query(0, 0, 1, 1) == 5
    assert st2d_max.query(2, 2, 3, 3) == 15
    
    # Test edge cases
    single = [42]
    st_single = SparseTable(single, 'min')
    assert st_single.query(0, 0) == 42
    
    print("Tous les tests passes")


if __name__ == "__main__":
    test()

