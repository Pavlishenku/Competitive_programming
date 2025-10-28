"""
Fast I/O pour Competitive Programming

Description:
    Optimisations d'entrées/sorties pour Python en CP.
    Crucial pour passer les contraintes de temps sur gros inputs.

Cas d'usage:
    - Problèmes avec beaucoup d'I/O
    - Lectures/écritures massives
    - Optimisation du temps d'exécution

Implémentation par: 2025-10-27
"""

import sys
from io import BytesIO, IOBase


# FastIO optimisé pour compétitions
BUFSIZE = 8192


class FastIO(IOBase):
    """Fast I/O pour compétitions avec gros volumes de données"""
    newlines = 0

    def __init__(self, file):
        self._fd = file.fileno()
        self.buffer = BytesIO()
        self.writable = "x" in file.mode or "r" not in file.mode
        self.write = self.buffer.write if self.writable else None

    def read(self):
        while True:
            b = os.read(self._fd, max(os.fstat(self._fd).st_size, BUFSIZE))
            if not b:
                break
            ptr = self.buffer.tell()
            self.buffer.seek(0, 2), self.buffer.write(b), self.buffer.seek(ptr)
        self.newlines = 0
        return self.buffer.read()

    def readline(self):
        while self.newlines == 0:
            b = os.read(self._fd, max(os.fstat(self._fd).st_size, BUFSIZE))
            self.newlines = b.count(b"\n") + (not b)
            ptr = self.buffer.tell()
            self.buffer.seek(0, 2), self.buffer.write(b), self.buffer.seek(ptr)
        self.newlines -= 1
        return self.buffer.readline()

    def flush(self):
        if self.writable:
            os.write(self._fd, self.buffer.getvalue())
            self.buffer.truncate(0), self.buffer.seek(0)


# Configuration I/O rapide simple
def setup_fast_io():
    """
    Configure l'I/O rapide pour competitive programming.
    À appeler au début du programme.
    """
    sys.stdin = open(0, 'r')
    sys.stdout = open(1, 'w')


# Fonctions helper pour I/O
def read_int():
    """Lit un entier"""
    return int(sys.stdin.readline())


def read_ints():
    """Lit plusieurs entiers sur une ligne"""
    return list(map(int, sys.stdin.readline().split()))


def read_str():
    """Lit une chaîne (sans le \\n)"""
    return sys.stdin.readline().strip()


def read_strs():
    """Lit plusieurs chaînes sur une ligne"""
    return sys.stdin.readline().split()


def print_list(lst, sep=' '):
    """Affiche une liste avec séparateur"""
    print(sep.join(map(str, lst)))


def print_matrix(matrix):
    """Affiche une matrice"""
    for row in matrix:
        print_list(row)


# Alternative: utiliser input remappé
def setup_fast_input():
    """
    Remplace input() par une version plus rapide.
    Utilisation: appeler au début, puis utiliser input() normalement.
    """
    import sys
    input = sys.stdin.readline
    return input


# Template de base pour CP
TEMPLATE_BASIC = '''
import sys
input = sys.stdin.readline

def solve():
    # Votre code ici
    pass

if __name__ == "__main__":
    solve()
'''


TEMPLATE_MULTIPLE_TESTS = '''
import sys
input = sys.stdin.readline

def solve():
    # Résoudre un cas de test
    pass

if __name__ == "__main__":
    t = int(input())
    for _ in range(t):
        solve()
'''


def test():
    """Tests basiques"""
    # Test read functions
    import io
    
    # Simuler stdin
    test_input = "42\\n1 2 3 4 5\\nhello world\\n"
    sys.stdin = io.StringIO(test_input)
    
    assert read_int() == 42
    assert read_ints() == [1, 2, 3, 4, 5]
    assert read_str() == "hello world"
    
    print("Tests I/O passes")


if __name__ == "__main__":
    test()

