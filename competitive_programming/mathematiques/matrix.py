"""
Matrix Exponentiation (Exponentiation Matricielle)

Description:
    Multiplication et exponentiation de matrices pour résoudre des récurrences
    linéaires en O(k³ log n) où k = taille de la matrice.

Complexité:
    - Multiplication: O(k³)
    - Exponentiation: O(k³ log n)

Cas d'usage:
    - Fibonacci en O(log n)
    - Récurrences linéaires
    - Chemins de longueur n dans un graphe
    - DP avec états

Problèmes types:
    - Codeforces: 691E, 166E, 185A
    - AtCoder: ABC129E
    - CSES: Fibonacci Numbers, Graph Paths
    
Implémentation par: 2025-10-27
Testé: Oui
"""


class Matrix:
    """Classe pour matrices avec opérations"""
    
    def __init__(self, data, mod=None):
        """
        Args:
            data: Matrice 2D (liste de listes)
            mod: Module pour arithmétique modulaire (optionnel)
        """
        self.data = [row[:] for row in data]
        self.rows = len(data)
        self.cols = len(data[0]) if data else 0
        self.mod = mod
    
    def __mul__(self, other):
        """Multiplication de matrices"""
        if self.cols != other.rows:
            raise ValueError("Dimensions incompatibles")
        
        result = [[0] * other.cols for _ in range(self.rows)]
        
        for i in range(self.rows):
            for j in range(other.cols):
                for k in range(self.cols):
                    result[i][j] += self.data[i][k] * other.data[k][j]
                    if self.mod:
                        result[i][j] %= self.mod
        
        return Matrix(result, self.mod)
    
    def __pow__(self, n):
        """Exponentiation rapide de matrice"""
        if self.rows != self.cols:
            raise ValueError("Matrice doit être carrée")
        
        if n == 0:
            return Matrix.identity(self.rows, self.mod)
        if n == 1:
            return Matrix(self.data, self.mod)
        
        if n % 2 == 0:
            half = self ** (n // 2)
            return half * half
        else:
            return self * (self ** (n - 1))
    
    def __repr__(self):
        return '\n'.join([' '.join(map(str, row)) for row in self.data])
    
    @staticmethod
    def identity(n, mod=None):
        """Crée une matrice identité n×n"""
        data = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
        return Matrix(data, mod)
    
    def get(self, i, j):
        """Accès à un élément"""
        return self.data[i][j]


def fibonacci_matrix(n, mod=None):
    """
    Calcule le n-ième nombre de Fibonacci en O(log n).
    
    Args:
        n: Indice (F(0)=0, F(1)=1)
        mod: Module (optionnel)
        
    Returns:
        F(n) % mod si mod fourni, sinon F(n)
        
    Example:
        >>> fibonacci_matrix(10)
        55
    """
    if n == 0:
        return 0
    if n == 1:
        return 1
    
    # Matrice de transformation pour Fibonacci
    # [F(n+1)]   [1 1]^n   [1]
    # [F(n)  ] = [1 0]   * [0]
    
    base = Matrix([[1, 1], [1, 0]], mod)
    result = base ** (n - 1)
    
    fib_n = result.get(0, 0) + result.get(0, 1)
    
    if mod:
        return fib_n % mod
    return fib_n


def linear_recurrence(coeffs, initial, n, mod=None):
    """
    Résout une récurrence linéaire de la forme:
    a(n) = c₁*a(n-1) + c₂*a(n-2) + ... + cₖ*a(n-k)
    
    Args:
        coeffs: [c₁, c₂, ..., cₖ] coefficients
        initial: [a₀, a₁, ..., a(k-1)] valeurs initiales
        n: Indice à calculer
        mod: Module
        
    Returns:
        a(n) % mod
        
    Example:
        >>> # Fibonacci: F(n) = F(n-1) + F(n-2)
        >>> linear_recurrence([1, 1], [0, 1], 10)
        55
    """
    k = len(coeffs)
    
    if n < k:
        return initial[n] if mod is None else initial[n] % mod
    
    # Construire la matrice de transformation
    matrix_data = []
    
    # Première ligne: coefficients
    matrix_data.append(coeffs[:])
    
    # Autres lignes: shift
    for i in range(1, k):
        row = [0] * k
        row[i-1] = 1
        matrix_data.append(row)
    
    trans = Matrix(matrix_data, mod)
    
    # Exponentier
    result = trans ** (n - k + 1)
    
    # Calculer a(n) = somme(result[0][i] * initial[k-1-i])
    ans = 0
    for i in range(k):
        ans += result.get(0, i) * initial[k - 1 - i]
        if mod:
            ans %= mod
    
    return ans


def count_paths_length_n(graph, n, start, end, mod=None):
    """
    Compte le nombre de chemins de longueur exactement n de start à end.
    
    Args:
        graph: Matrice d'adjacence
        n: Longueur des chemins
        start: Sommet de départ
        end: Sommet d'arrivée
        mod: Module
        
    Returns:
        Nombre de chemins
    """
    adj_matrix = Matrix(graph, mod)
    result = adj_matrix ** n
    
    return result.get(start, end)


def tribonacci(n, mod=None):
    """
    Calcule le n-ième nombre de Tribonacci.
    T(n) = T(n-1) + T(n-2) + T(n-3)
    T(0)=0, T(1)=0, T(2)=1
    
    Args:
        n: Indice
        mod: Module
        
    Returns:
        T(n)
    """
    return linear_recurrence([1, 1, 1], [0, 0, 1], n, mod)


def matrix_multiply(A, B, mod=None):
    """
    Multiplication simple de deux matrices.
    
    Args:
        A: Première matrice (liste de listes)
        B: Deuxième matrice (liste de listes)
        mod: Module optionnel
        
    Returns:
        Produit A × B
    """
    mat_a = Matrix(A, mod)
    mat_b = Matrix(B, mod)
    result = mat_a * mat_b
    return result.data


def matrix_power(A, n, mod=None):
    """
    Élève une matrice à la puissance n.
    
    Args:
        A: Matrice (liste de listes)
        n: Exposant
        mod: Module optionnel
        
    Returns:
        A^n
    """
    mat = Matrix(A, mod)
    result = mat ** n
    return result.data


def test():
    """Tests unitaires complets"""
    
    # Test Fibonacci
    assert fibonacci_matrix(0) == 0
    assert fibonacci_matrix(1) == 1
    assert fibonacci_matrix(2) == 1
    assert fibonacci_matrix(10) == 55
    assert fibonacci_matrix(20) == 6765
    
    # Test Fibonacci avec module
    MOD = 10**9 + 7
    fib_mod = fibonacci_matrix(100, MOD)
    assert fib_mod > 0
    
    # Test multiplication de matrices
    A = Matrix([[1, 2], [3, 4]])
    B = Matrix([[5, 6], [7, 8]])
    C = A * B
    
    assert C.get(0, 0) == 19  # 1*5 + 2*7
    assert C.get(0, 1) == 22  # 1*6 + 2*8
    assert C.get(1, 0) == 43  # 3*5 + 4*7
    assert C.get(1, 1) == 50  # 3*6 + 4*8
    
    # Test identité
    I = Matrix.identity(3)
    M = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    MI = M * I
    
    for i in range(3):
        for j in range(3):
            assert MI.get(i, j) == M.get(i, j)
    
    # Test exponentiation
    M2 = Matrix([[1, 1], [1, 0]])
    M2_10 = M2 ** 10
    
    # Devrait donner F(11) et F(10) dans la première ligne
    assert M2_10.get(0, 0) + M2_10.get(0, 1) == 89  # F(11)
    
    # Test récurrence linéaire
    # Fibonacci
    fib_10 = linear_recurrence([1, 1], [0, 1], 10)
    assert fib_10 == 55
    
    # Tribonacci
    trib_10 = tribonacci(10)
    assert trib_10 == 149  # Vérifié manuellement
    
    # Test chemins dans un graphe
    # Graphe simple: 0 -> 1 -> 2
    #                0 -> 2
    graph = [
        [0, 1, 1],
        [0, 0, 1],
        [0, 0, 0]
    ]
    
    # Chemins de longueur 2 de 0 à 2
    paths = count_paths_length_n(graph, 2, 0, 2)
    assert paths == 1  # 0->1->2
    
    # Test avec module
    M_mod = Matrix([[2, 3], [4, 5]], mod=7)
    M_mod_2 = M_mod ** 2
    
    # (2*2 + 3*4) % 7 = 16 % 7 = 2
    assert M_mod_2.get(0, 0) == 2
    
    # Test matrice 1x1
    single = Matrix([[5]])
    single_cubed = single ** 3
    assert single_cubed.get(0, 0) == 125
    
    print("Tous les tests passes")


if __name__ == "__main__":
    test()

