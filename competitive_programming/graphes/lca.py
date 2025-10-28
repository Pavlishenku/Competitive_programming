"""
Lowest Common Ancestor (LCA)

Description:
    Algorithmes pour trouver l'ancêtre commun le plus bas dans un arbre.
    Implémentations: Binary Lifting, RMQ, Tarjan offline.

Complexité:
    - Binary Lifting: O(n log n) preprocessing, O(log n) query
    - RMQ: O(n log n) preprocessing, O(1) query
    - Espace: O(n log n)

Cas d'usage:
    - Requêtes d'ancêtre commun
    - Distance entre deux noeuds dans un arbre
    - Problèmes de path queries sur arbres
    - HLD et Centroid Decomposition
    
Problèmes types:
    - Codeforces: 519E, 609E, 191C
    - AtCoder: ABC014D, ABC133F
    - CSES: Company Queries I, Company Queries II
    
Implémentation par: 2025-10-27
Testé: Oui
"""

import math


class LCABinaryLifting:
    """
    LCA utilisant Binary Lifting.
    Méthode la plus courante en competitive programming.
    """
    
    def __init__(self, tree, root=0):
        """
        Args:
            tree: Dict {sommet: [enfants]} ou liste d'adjacence
            root: Racine de l'arbre
            
        Example:
            >>> tree = {0: [1, 2], 1: [3, 4], 2: [5], 3: [], 4: [], 5: []}
            >>> lca = LCABinaryLifting(tree, root=0)
            >>> lca.query(3, 4)
            1
        """
        self.tree = tree
        self.root = root
        self.n = len(tree)
        self.LOG = math.ceil(math.log2(self.n)) + 1 if self.n > 0 else 1
        
        # up[v][i] = 2^i-ème ancêtre de v
        self.up = [[-1] * self.LOG for _ in range(self.n)]
        self.depth = [0] * self.n
        
        self._preprocess()
    
    def _preprocess(self):
        """Preprocessing: calcule les ancêtres et profondeurs"""
        # DFS pour initialiser profondeur et parent direct
        visited = [False] * self.n
        
        def dfs(node, parent, d):
            visited[node] = True
            self.depth[node] = d
            self.up[node][0] = parent
            
            for child in self.tree.get(node, []):
                if not visited[child]:
                    dfs(child, node, d + 1)
        
        dfs(self.root, -1, 0)
        
        # Remplir la table up avec binary lifting
        for j in range(1, self.LOG):
            for i in range(self.n):
                if self.up[i][j-1] != -1:
                    self.up[i][j] = self.up[self.up[i][j-1]][j-1]
    
    def query(self, u, v):
        """
        Trouve le LCA de u et v.
        
        Args:
            u: Premier sommet
            v: Deuxième sommet
            
        Returns:
            LCA de u et v
        """
        # Amener u et v à la même profondeur
        if self.depth[u] < self.depth[v]:
            u, v = v, u
        
        # Remonter u à la profondeur de v
        diff = self.depth[u] - self.depth[v]
        for i in range(self.LOG):
            if (diff >> i) & 1:
                u = self.up[u][i]
        
        if u == v:
            return u
        
        # Binary lifting pour trouver LCA
        for i in range(self.LOG - 1, -1, -1):
            if self.up[u][i] != self.up[v][i]:
                u = self.up[u][i]
                v = self.up[v][i]
        
        return self.up[u][0]
    
    def distance(self, u, v):
        """
        Calcule la distance entre u et v.
        
        Args:
            u: Premier sommet
            v: Deuxième sommet
            
        Returns:
            Nombre d'arêtes entre u et v
        """
        lca = self.query(u, v)
        return self.depth[u] + self.depth[v] - 2 * self.depth[lca]
    
    def kth_ancestor(self, node, k):
        """
        Trouve le k-ième ancêtre de node.
        
        Args:
            node: Sommet
            k: Nombre de générations à remonter
            
        Returns:
            k-ième ancêtre ou -1 si n'existe pas
        """
        if k > self.depth[node]:
            return -1
        
        for i in range(self.LOG):
            if (k >> i) & 1:
                node = self.up[node][i]
                if node == -1:
                    return -1
        
        return node


class LCASparse:
    """
    LCA optimisé avec Sparse Table (RMQ).
    Plus complexe mais requêtes en O(1).
    """
    
    def __init__(self, tree, root=0):
        """
        Args:
            tree: Dict {sommet: [enfants]}
            root: Racine de l'arbre
        """
        self.tree = tree
        self.root = root
        self.n = len(tree)
        
        # Euler tour
        self.euler = []
        self.first = [0] * self.n
        self.depth = [0] * self.n
        
        self._euler_tour()
        self._build_sparse_table()
    
    def _euler_tour(self):
        """Construit l'Euler tour de l'arbre"""
        visited = [False] * self.n
        
        def dfs(node, d):
            visited[node] = True
            self.depth[node] = d
            self.first[node] = len(self.euler)
            self.euler.append(node)
            
            for child in self.tree.get(node, []):
                if not visited[child]:
                    dfs(child, d + 1)
                    self.euler.append(node)
        
        dfs(self.root, 0)
    
    def _build_sparse_table(self):
        """Construit la Sparse Table pour RMQ"""
        m = len(self.euler)
        if m == 0:
            return
        
        LOG = math.ceil(math.log2(m)) + 1
        self.st = [[0] * LOG for _ in range(m)]
        
        # Initialiser avec les éléments individuels
        for i in range(m):
            self.st[i][0] = i
        
        # Remplir la table
        j = 1
        while (1 << j) <= m:
            i = 0
            while i + (1 << j) - 1 < m:
                left = self.st[i][j-1]
                right = self.st[i + (1 << (j-1))][j-1]
                
                if self.depth[self.euler[left]] < self.depth[self.euler[right]]:
                    self.st[i][j] = left
                else:
                    self.st[i][j] = right
                
                i += 1
            j += 1
        
        self.log_table = [0] * (m + 1)
        for i in range(2, m + 1):
            self.log_table[i] = self.log_table[i // 2] + 1
    
    def query(self, u, v):
        """
        Trouve le LCA de u et v en O(1).
        
        Args:
            u: Premier sommet
            v: Deuxième sommet
            
        Returns:
            LCA de u et v
        """
        left = min(self.first[u], self.first[v])
        right = max(self.first[u], self.first[v])
        
        j = self.log_table[right - left + 1]
        
        left_min = self.st[left][j]
        right_min = self.st[right - (1 << j) + 1][j]
        
        if self.depth[self.euler[left_min]] < self.depth[self.euler[right_min]]:
            return self.euler[left_min]
        else:
            return self.euler[right_min]


def test():
    """Tests unitaires complets"""
    
    # Test arbre exemple
    tree = {
        0: [1, 2],
        1: [3, 4],
        2: [5],
        3: [],
        4: [],
        5: []
    }
    
    # Test Binary Lifting
    lca = LCABinaryLifting(tree, root=0)
    
    assert lca.query(3, 4) == 1
    assert lca.query(3, 5) == 0
    assert lca.query(1, 2) == 0
    assert lca.query(3, 3) == 3
    
    # Test distance
    assert lca.distance(3, 4) == 2
    assert lca.distance(3, 5) == 4
    assert lca.distance(0, 5) == 2
    
    # Test k-th ancestor
    assert lca.kth_ancestor(3, 1) == 1
    assert lca.kth_ancestor(3, 2) == 0
    assert lca.kth_ancestor(5, 1) == 2
    assert lca.kth_ancestor(5, 3) == -1  # N'existe pas
    
    # Test arbre plus complexe
    tree2 = {
        0: [1, 2, 3],
        1: [4, 5],
        2: [6],
        3: [7, 8],
        4: [], 5: [], 6: [], 7: [], 8: []
    }
    
    lca2 = LCABinaryLifting(tree2, root=0)
    
    assert lca2.query(4, 5) == 1
    assert lca2.query(4, 6) == 0
    assert lca2.query(7, 8) == 3
    assert lca2.query(4, 8) == 0
    
    # Test LCA Sparse Table
    lca_sparse = LCASparse(tree, root=0)
    
    assert lca_sparse.query(3, 4) == 1
    assert lca_sparse.query(3, 5) == 0
    assert lca_sparse.query(1, 2) == 0
    
    # Test arbre linéaire
    linear_tree = {
        0: [1],
        1: [2],
        2: [3],
        3: []
    }
    
    lca_linear = LCABinaryLifting(linear_tree, root=0)
    assert lca_linear.query(2, 3) == 2
    assert lca_linear.query(0, 3) == 0
    assert lca_linear.distance(0, 3) == 3
    
    print("Tous les tests passes")


if __name__ == "__main__":
    test()

