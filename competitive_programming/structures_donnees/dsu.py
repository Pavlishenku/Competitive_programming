"""
Disjoint Set Union (DSU) / Union-Find

Description:
    Structure de données pour gérer des ensembles disjoints avec opérations
    d'union et de recherche. Utilise path compression et union by rank
    pour des performances quasi-constantes.

Complexité:
    - Temps: O(α(n)) par opération (α = inverse d'Ackermann, quasi-constant)
    - Espace: O(n)

Cas d'usage:
    - Détection de composantes connexes
    - Algorithmes MST (Kruskal)
    - Détection de cycles dans graphes non-orientés
    - Problèmes de connectivité dynamique
    
Problèmes types:
    - Codeforces: 277A, 25D, 1176E
    - AtCoder: ABC120D, ABC157D
    - USACO: Closing the Farm
    
Implémentation par: 2025-10-27
Testé: Oui
"""

class DSU:
    """
    Disjoint Set Union avec path compression et union by rank.
    """
    
    def __init__(self, n):
        """
        Initialise DSU avec n éléments (0 à n-1).
        
        Args:
            n: Nombre d'éléments
            
        Example:
            >>> dsu = DSU(5)
            >>> dsu.union(0, 1)
            >>> dsu.find(0) == dsu.find(1)
            True
        """
        self.parent = list(range(n))
        self.rank = [0] * n
        self.size = [1] * n
        self.num_components = n
    
    def find(self, x):
        """
        Trouve le représentant (racine) de l'ensemble contenant x.
        Utilise path compression pour optimisation.
        
        Args:
            x: Élément à rechercher
            
        Returns:
            Représentant de l'ensemble contenant x
        """
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]
    
    def union(self, x, y):
        """
        Fusionne les ensembles contenant x et y.
        Utilise union by rank pour garder l'arbre plat.
        
        Args:
            x: Premier élément
            y: Deuxième élément
            
        Returns:
            True si fusion effectuée, False si déjà dans le même ensemble
        """
        root_x = self.find(x)
        root_y = self.find(y)
        
        if root_x == root_y:
            return False
        
        # Union by rank
        if self.rank[root_x] < self.rank[root_y]:
            root_x, root_y = root_y, root_x
        
        self.parent[root_y] = root_x
        self.size[root_x] += self.size[root_y]
        
        if self.rank[root_x] == self.rank[root_y]:
            self.rank[root_x] += 1
        
        self.num_components -= 1
        return True
    
    def same(self, x, y):
        """
        Vérifie si x et y sont dans le même ensemble.
        
        Args:
            x: Premier élément
            y: Deuxième élément
            
        Returns:
            True si dans le même ensemble, False sinon
        """
        return self.find(x) == self.find(y)
    
    def get_size(self, x):
        """
        Retourne la taille de l'ensemble contenant x.
        
        Args:
            x: Élément
            
        Returns:
            Taille de l'ensemble
        """
        return self.size[self.find(x)]
    
    def get_num_components(self):
        """
        Retourne le nombre de composantes connexes.
        
        Returns:
            Nombre d'ensembles disjoints
        """
        return self.num_components


class UnionFind(DSU):
    """Alias pour DSU"""
    pass


def test():
    """Tests unitaires complets"""
    
    # Test basique d'union et find
    dsu = DSU(5)
    assert dsu.get_num_components() == 5
    
    dsu.union(0, 1)
    assert dsu.same(0, 1)
    assert not dsu.same(0, 2)
    assert dsu.get_num_components() == 4
    
    # Test de taille
    assert dsu.get_size(0) == 2
    assert dsu.get_size(2) == 1
    
    # Test d'unions multiples
    dsu.union(2, 3)
    dsu.union(3, 4)
    assert dsu.same(2, 4)
    assert dsu.get_size(2) == 3
    assert dsu.get_num_components() == 2
    
    # Test union redondante
    result = dsu.union(0, 1)
    assert not result  # Déjà dans le même ensemble
    assert dsu.get_num_components() == 2
    
    # Test union finale
    dsu.union(1, 4)
    assert dsu.same(0, 4)
    assert dsu.get_num_components() == 1
    assert dsu.get_size(0) == 5
    
    # Test avec graphe exemple (détection de cycles)
    dsu2 = DSU(4)
    edges = [(0, 1), (1, 2), (0, 3)]
    for u, v in edges:
        assert not dsu2.same(u, v)  # Pas de cycle
        dsu2.union(u, v)
    
    # Cette arête créerait un cycle
    assert dsu2.same(2, 3)
    
    print("Tous les tests passes")


if __name__ == "__main__":
    test()

