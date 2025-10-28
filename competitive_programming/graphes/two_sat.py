"""
================================================================================
2-SAT (2-Satisfiability)
================================================================================

Description:
-----------
Resout le probleme 2-SAT: etant donne une formule booleenne en CNF avec
exactement 2 litteraux par clause, determine si elle est satisfiable.
Utilise les composantes fortement connexes (Tarjan/Kosaraju).

Complexite:
-----------
- Construction: O(n + m) ou n = variables, m = clauses
- Resolution: O(n + m)
- Espace: O(n + m)

Cas d'usage typiques:
--------------------
1. Contraintes booleennes avec implications
2. Problemes de coloration avec contraintes
3. Scheduling avec conflicts
4. Circuit design

Problemes classiques:
--------------------
- Codeforces 228E - The Road to Berland is Paved With Good Intentions
- SPOJ MAXOR - Maximum OR
- USACO Training - Betsy's Tour
- AtCoder ABC 187F - Close Group

Auteur: Assistant CP
Date: 2025
================================================================================
"""

from typing import List, Tuple, Optional
from collections import defaultdict, deque


class TwoSAT:
    """
    2-SAT Solver utilisant SCC.
    
    Exemple:
    --------
    >>> sat = TwoSAT(3)  # 3 variables (0, 1, 2)
    >>> 
    >>> # Clause: (x0 OR x1) - au moins un vrai
    >>> sat.add_clause(0, True, 1, True)
    >>> 
    >>> # Clause: (NOT x0 OR x2) - si x0 alors x2
    >>> sat.add_clause(0, False, 2, True)
    >>> 
    >>> # Clause: (NOT x1 OR NOT x2) - pas les deux vrais
    >>> sat.add_clause(1, False, 2, False)
    >>> 
    >>> is_sat, assignment = sat.solve()
    >>> if is_sat:
    ...     print(f"Satisfiable: {assignment}")
    """
    
    def __init__(self, n: int):
        """
        Args:
            n: Nombre de variables (indexees 0 a n-1)
        """
        self.n = n
        self.graph = defaultdict(list)  # Graphe d'implication
        self.clauses = []  # Pour debug
    
    def _var(self, x: int, value: bool) -> int:
        """
        Convertit (variable, valeur) en noeud du graphe.
        Noeud 2*x = x est vrai
        Noeud 2*x+1 = x est faux
        """
        return 2 * x if value else 2 * x + 1
    
    def _neg(self, node: int) -> int:
        """Retourne la negation d'un noeud"""
        return node ^ 1
    
    def add_clause(self, x: int, x_val: bool, y: int, y_val: bool):
        """
        Ajoute une clause (x=x_val OR y=y_val).
        
        Equivalent a deux implications:
        - NOT x_val => y=y_val
        - NOT y_val => x=x_val
        
        Args:
            x, y: Variables
            x_val, y_val: Valeurs booleennes
        """
        self.clauses.append((x, x_val, y, y_val))
        
        # (x_val OR y_val) <=> (NOT x_val => y_val) AND (NOT y_val => x_val)
        node_x = self._var(x, x_val)
        node_y = self._var(y, y_val)
        
        self.graph[self._neg(node_x)].append(node_y)
        self.graph[self._neg(node_y)].append(node_x)
    
    def add_implication(self, x: int, x_val: bool, y: int, y_val: bool):
        """
        Ajoute une implication: (x=x_val => y=y_val).
        
        Equivalent a la clause (NOT x_val OR y_val).
        """
        self.add_clause(x, not x_val, y, y_val)
    
    def set_value(self, x: int, value: bool):
        """
        Force une variable a avoir une valeur specifique.
        
        Equivalent a la clause (x=value OR x=value).
        """
        self.add_clause(x, value, x, value)
    
    def _tarjan_scc(self) -> List[int]:
        """
        Trouve les composantes fortement connexes avec Tarjan.
        
        Returns:
            Liste comp[node] = id de composante
        """
        n_nodes = 2 * self.n
        
        index = [None] * n_nodes
        lowlink = [None] * n_nodes
        on_stack = [False] * n_nodes
        stack = []
        scc_id = [-1] * n_nodes
        current_index = [0]
        current_scc = [0]
        
        def strongconnect(v: int):
            index[v] = current_index[0]
            lowlink[v] = current_index[0]
            current_index[0] += 1
            stack.append(v)
            on_stack[v] = True
            
            for w in self.graph[v]:
                if index[w] is None:
                    strongconnect(w)
                    lowlink[v] = min(lowlink[v], lowlink[w])
                elif on_stack[w]:
                    lowlink[v] = min(lowlink[v], index[w])
            
            if lowlink[v] == index[v]:
                # Nouvelle SCC
                while True:
                    w = stack.pop()
                    on_stack[w] = False
                    scc_id[w] = current_scc[0]
                    if w == v:
                        break
                current_scc[0] += 1
        
        for v in range(n_nodes):
            if index[v] is None and (v in self.graph or any(v in adj for adj in self.graph.values())):
                strongconnect(v)
        
        return scc_id
    
    def solve(self) -> Tuple[bool, Optional[List[bool]]]:
        """
        Resout le 2-SAT.
        
        Returns:
            (is_satisfiable, assignment)
            Si satisfiable, assignment[i] = valeur de la variable i
            Sinon, assignment = None
        """
        # Trouve SCC
        scc_id = self._tarjan_scc()
        
        # Verifie satisfiabilite
        for i in range(self.n):
            node_true = self._var(i, True)
            node_false = self._var(i, False)
            
            if scc_id[node_true] == scc_id[node_false]:
                # Contradiction: x et NOT x dans la meme SCC
                return False, None
        
        # Construit assignation
        # Si SCC(x) > SCC(NOT x), alors x = TRUE
        # (ordre topologique inverse)
        assignment = []
        for i in range(self.n):
            node_true = self._var(i, True)
            node_false = self._var(i, False)
            
            # SCC avec id plus petit est plus tard dans l'ordre topo
            # On veut assigner TRUE si possible (preference)
            assignment.append(scc_id[node_true] < scc_id[node_false])
        
        return True, assignment


def solve_2sat_simple(n: int, clauses: List[Tuple[int, bool, int, bool]]) -> Optional[List[bool]]:
    """
    Interface simplifiee pour 2-SAT.
    
    Args:
        n: Nombre de variables
        clauses: Liste de (x, x_val, y, y_val) pour (x=x_val OR y=y_val)
        
    Returns:
        Assignment si satisfiable, None sinon
        
    Exemple:
    --------
    >>> clauses = [
    ...     (0, True, 1, True),     # x0 OR x1
    ...     (0, False, 2, True),    # NOT x0 OR x2
    ...     (1, False, 2, False)    # NOT x1 OR NOT x2
    ... ]
    >>> result = solve_2sat_simple(3, clauses)
    >>> print(result)  # [False, True, False] ou autre solution valide
    """
    sat = TwoSAT(n)
    
    for x, x_val, y, y_val in clauses:
        sat.add_clause(x, x_val, y, y_val)
    
    is_sat, assignment = sat.solve()
    return assignment if is_sat else None


# ============================================================================
# TESTS UNITAIRES
# ============================================================================

def test_2sat_basic():
    """Test basic 2-SAT"""
    sat = TwoSAT(2)
    
    # (x0 OR x1) AND (NOT x0 OR NOT x1)
    # Solution: exactement un vrai
    sat.add_clause(0, True, 1, True)
    sat.add_clause(0, False, 1, False)
    
    is_sat, assignment = sat.solve()
    
    assert is_sat
    assert assignment[0] != assignment[1]  # Exactement un vrai
    
    print("✓ Test 2-SAT basic passed")


def test_2sat_unsatisfiable():
    """Test unsatisfiable 2-SAT"""
    sat = TwoSAT(1)
    
    # x0 AND NOT x0 - impossible
    sat.set_value(0, True)
    sat.set_value(0, False)
    
    is_sat, _ = sat.solve()
    
    assert not is_sat
    
    print("✓ Test 2-SAT unsatisfiable passed")


def test_2sat_implications():
    """Test implications"""
    sat = TwoSAT(3)
    
    # x0 => x1
    sat.add_implication(0, True, 1, True)
    
    # x1 => x2
    sat.add_implication(1, True, 2, True)
    
    # x0 = True
    sat.set_value(0, True)
    
    is_sat, assignment = sat.solve()
    
    assert is_sat
    # Si x0=True, alors x1=True et x2=True
    if assignment[0]:
        assert assignment[1] and assignment[2]
    
    print("✓ Test 2-SAT implications passed")


def test_2sat_complex():
    """Test probleme complexe"""
    sat = TwoSAT(4)
    
    # (x0 OR x1) AND (NOT x0 OR x2) AND (NOT x1 OR x3) AND (NOT x2 OR NOT x3)
    sat.add_clause(0, True, 1, True)
    sat.add_clause(0, False, 2, True)
    sat.add_clause(1, False, 3, True)
    sat.add_clause(2, False, 3, False)
    
    is_sat, assignment = sat.solve()
    
    assert is_sat
    
    # Verifie les clauses
    assert assignment[0] or assignment[1]
    assert not assignment[0] or assignment[2]
    assert not assignment[1] or assignment[3]
    assert not (assignment[2] and assignment[3])
    
    print("✓ Test 2-SAT complex passed")


def test_solve_2sat_simple():
    """Test interface simplifiee"""
    clauses = [
        (0, True, 1, True),
        (0, False, 2, True),
        (1, False, 2, False)
    ]
    
    result = solve_2sat_simple(3, clauses)
    
    assert result is not None
    assert len(result) == 3
    
    print("✓ Test solve 2-SAT simple passed")


# ============================================================================
# BENCHMARK
# ============================================================================

def benchmark_2sat():
    """Benchmark 2-SAT"""
    import time
    import random
    
    print("\n=== Benchmark 2-SAT ===")
    
    for n in [100, 500, 1000, 2000]:
        sat = TwoSAT(n)
        
        # Genere clauses aleatoires
        num_clauses = n * 3
        for _ in range(num_clauses):
            x = random.randint(0, n-1)
            y = random.randint(0, n-1)
            x_val = random.choice([True, False])
            y_val = random.choice([True, False])
            sat.add_clause(x, x_val, y, y_val)
        
        start = time.time()
        is_sat, _ = sat.solve()
        elapsed = time.time() - start
        
        print(f"n={n:4d}, clauses={num_clauses}: {elapsed*1000:6.2f}ms, SAT={is_sat}")


if __name__ == "__main__":
    test_2sat_basic()
    test_2sat_unsatisfiable()
    test_2sat_implications()
    test_2sat_complex()
    test_solve_2sat_simple()
    
    benchmark_2sat()
    
    print("\n✓ Tous les tests 2-SAT passes!")

