"""
Bibliothèque Python pour Programmation Compétitive

Package complet d'algorithmes et structures de données optimisés 
pour les olympiades informatiques et competitive programming.

Modules:
    - structures_donnees: Segment Tree, Fenwick Tree, DSU, Trie, etc.
    - graphes: DFS, BFS, Dijkstra, MST, Max Flow, etc.
    - mathematiques: Nombres premiers, modular arithmetic, combinatoire
    - strings: KMP, Z-algorithm, rolling hash, etc.
    - geometrie: Convex hull, intersections, calculs géométriques
    - dynamique: DP classiques et optimisations avancées
    - recherche: Binary search, ternary search, etc.
    - utils: FastIO, templates, debugging helpers
"""

__version__ = "1.0.0"
__author__ = "Competitive Programming Library"

# Imports des modules principaux
from . import structures_donnees
from . import graphes
from . import mathematiques
from . import strings
from . import geometrie
from . import dynamique
from . import recherche
from . import utils

__all__ = [
    'structures_donnees',
    'graphes',
    'mathematiques',
    'strings',
    'geometrie',
    'dynamique',
    'recherche',
    'utils',
]

