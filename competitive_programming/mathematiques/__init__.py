"""
Algorithmes mathématiques pour programmation compétitive
"""

from .primes import (
    sieve_of_eratosthenes, prime_factorization, is_prime,
    miller_rabin, euler_phi
)
from .modular import (
    pow_mod, mod_inverse, ModularFactorial,
    add_mod, sub_mod, mul_mod, div_mod
)
from .matrix import Matrix, fibonacci_matrix, linear_recurrence
from .fft import fft, multiply_polynomials, convolve, NTT
from .pollard_rho import (
    pollard_rho, prime_factorization as pollard_factorization,
    factorize, get_divisors, count_divisors, sum_of_divisors, euler_phi_fast
)

__all__ = [
    'sieve_of_eratosthenes', 'prime_factorization', 'is_prime',
    'miller_rabin', 'euler_phi',
    'pow_mod', 'mod_inverse', 'ModularFactorial',
    'add_mod', 'sub_mod', 'mul_mod', 'div_mod',
    'Matrix', 'fibonacci_matrix', 'linear_recurrence',
    'fft', 'multiply_polynomials', 'convolve', 'NTT',
    'pollard_rho', 'pollard_factorization', 'factorize', 
    'get_divisors', 'count_divisors', 'sum_of_divisors', 'euler_phi_fast'
]
