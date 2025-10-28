"""
Structures de données pour programmation compétitive
"""

from .dsu import DSU, UnionFind
from .segment_tree import SegmentTree, SegmentTreeLazy
from .fenwick_tree import FenwickTree, FenwickTree2D, FenwickTreeRangeUpdate
from .trie import Trie, BinaryTrie
from .sparse_table import SparseTable, SparseTable2D
from .mo_algorithm import MoAlgorithm, MoDistinctElements, count_distinct_in_ranges
from .link_cut_tree import LinkCutTree, solve_dynamic_connectivity
from .splay_tree import SplayTree
from .treap import Treap, ImplicitTreap
from .persistent_segment_tree import PersistentSegmentTree, PersistentSegmentTreeKthNumber
from .wavelet_tree import WaveletTree

__all__ = [
    'DSU', 'UnionFind',
    'SegmentTree', 'SegmentTreeLazy',
    'FenwickTree', 'FenwickTree2D', 'FenwickTreeRangeUpdate',
    'Trie', 'BinaryTrie',
    'SparseTable', 'SparseTable2D',
    'MoAlgorithm', 'MoDistinctElements', 'count_distinct_in_ranges',
    'LinkCutTree', 'solve_dynamic_connectivity',
    'SplayTree',
    'Treap', 'ImplicitTreap',
    'PersistentSegmentTree', 'PersistentSegmentTreeKthNumber',
    'WaveletTree'
]

