# Competitive Programming Library

Bibliothèque complète d'algorithmes et structures de données pour programmation compétitive.

## Installation

```bash
git clone https://github.com/Pavlishenku/Competitive_programming.git
cd Competitive_programming
```

## Utilisation

```python
from competitive_programming.structures_donnees import DSU, SegmentTree
from competitive_programming.graphes import dijkstra, bfs
from competitive_programming.mathematiques import sieve_of_eratosthenes, pow_mod

# DSU
dsu = DSU(10)
dsu.union(1, 2)
print(dsu.same(1, 3))

# Segment Tree
st = SegmentTree([1, 3, 5, 7, 9], operation='sum')
print(st.query(1, 3))

# Dijkstra
graph = {0: [(1, 4), (2, 1)], 1: [(3, 1)], 2: [(1, 2), (3, 5)], 3: []}
distances = dijkstra(graph, 0)
```

## Contenu

### Structures de Données
DSU, Segment Tree, Fenwick Tree, Trie, Sparse Table, Mo's Algorithm, Link-Cut Tree, Splay Tree, Treap, Persistent Segment Tree, Wavelet Tree

### Graphes
DFS/BFS, Dijkstra, MST (Kruskal/Prim), Bellman-Ford, Floyd-Warshall, Tarjan, Topological Sort, LCA, Max Flow, Bipartite Matching, Centroid Decomposition, Heavy-Light Decomposition, Min Cost Max Flow, 2-SAT, Min Cut

### Mathématiques
Primes, Modular Arithmetic, Combinatorics, GCD/LCM, Chinese Remainder Theorem, Matrix Exponentiation, FFT/NTT, Pollard's Rho

### Strings
KMP, Z-Algorithm, Rabin-Karp, Manacher, String Hashing, Aho-Corasick, Suffix Array, Suffix Automaton, Lyndon Factorization, Burrows-Wheeler Transform

### Programmation Dynamique
Knapsack, LIS, LCS, Edit Distance, DP Bitmask, Digit DP, SOS DP, Convex Hull Trick, Aliens Trick, Knuth Optimization, Monotone Queue DP, Divide & Conquer DP

### Autres
Géométrie 2D, Binary Search, Fast I/O, Templates

## Exemples

35+ solutions de compétitions réelles (Google Code Jam, Kick Start, Meta Hacker Cup, AtCoder, CSES)

## License

MIT
