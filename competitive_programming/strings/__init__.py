"""
Algorithmes de traitement de chaînes pour programmation compétitive
"""

from .pattern_matching import kmp_search, z_algorithm, rabin_karp, longest_palindrome
from .string_hashing import StringHash, DoubleHash
from .aho_corasick import AhoCorasick, find_all_patterns
from .suffix_array import SuffixArray, kasai_lcp
from .suffix_automaton import SuffixAutomaton, count_distinct_substrings, longest_common_substring
from .advanced_strings import (
    lyndon_factorization, minimum_rotation,
    burrows_wheeler_transform, inverse_burrows_wheeler,
    run_length_encode, run_length_decode, advanced_rle_compress
)

__all__ = [
    'kmp_search', 'z_algorithm', 'rabin_karp', 'longest_palindrome',
    'StringHash', 'DoubleHash',
    'AhoCorasick', 'find_all_patterns',
    'SuffixArray', 'kasai_lcp',
    'SuffixAutomaton', 'count_distinct_substrings', 'longest_common_substring',
    'lyndon_factorization', 'minimum_rotation',
    'burrows_wheeler_transform', 'inverse_burrows_wheeler',
    'run_length_encode', 'run_length_decode', 'advanced_rle_compress'
]

