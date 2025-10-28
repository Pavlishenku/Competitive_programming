"""
Templates pour Competitive Programming

Description:
    Templates de code réutilisables pour démarrer rapidement.

Implémentation par: 2025-10-27
"""

# Template de base
TEMPLATE_BASIC = """
import sys
from collections import defaultdict, deque, Counter
from heapq import heappush, heappop
from bisect import bisect_left, bisect_right
import math

input = sys.stdin.readline

def solve():
    # Lire input
    n = int(input())
    arr = list(map(int, input().split()))
    
    # Résoudre le problème
    result = 0
    
    # Output
    print(result)

if __name__ == "__main__":
    solve()
"""


# Template avec multiples cas de test
TEMPLATE_MULTIPLE = """
import sys
input = sys.stdin.readline

def solve():
    # Un cas de test
    n = int(input())
    
    # Résoudre
    result = n
    
    print(result)

if __name__ == "__main__":
    t = int(input())
    for _ in range(t):
        solve()
"""


# Template avec graphes
TEMPLATE_GRAPH = """
import sys
from collections import defaultdict, deque

input = sys.stdin.readline

def solve():
    n, m = map(int, input().split())
    
    # Construire le graphe
    graph = defaultdict(list)
    for _ in range(m):
        u, v = map(int, input().split())
        graph[u].append(v)
        graph[v].append(u)
    
    # BFS/DFS
    visited = [False] * (n + 1)
    
    def bfs(start):
        queue = deque([start])
        visited[start] = True
        
        while queue:
            node = queue.popleft()
            for neighbor in graph[node]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)
    
    bfs(1)
    
    print("Done")

if __name__ == "__main__":
    solve()
"""


# Template avec DP
TEMPLATE_DP = """
import sys
input = sys.stdin.readline

def solve():
    n = int(input())
    arr = list(map(int, input().split()))
    
    # Initialiser DP
    dp = [0] * (n + 1)
    dp[0] = 0
    
    # Remplir DP
    for i in range(1, n + 1):
        dp[i] = dp[i-1] + arr[i-1]
    
    print(dp[n])

if __name__ == "__main__":
    solve()
"""


# Template avec modular arithmetic
TEMPLATE_MODULAR = """
import sys
input = sys.stdin.readline

MOD = 10**9 + 7

def pow_mod(base, exp, mod=MOD):
    result = 1
    base %= mod
    while exp > 0:
        if exp & 1:
            result = (result * base) % mod
        base = (base * base) % mod
        exp >>= 1
    return result

def mod_inverse(a, mod=MOD):
    return pow_mod(a, mod - 2, mod)

def solve():
    n = int(input())
    
    # Calculs modulaires
    result = pow_mod(2, n)
    
    print(result)

if __name__ == "__main__":
    solve()
"""


# Template avec binary search
TEMPLATE_BINARY_SEARCH = """
import sys
from bisect import bisect_left

input = sys.stdin.readline

def binary_search_answer(predicate, left, right):
    result = left - 1
    while left <= right:
        mid = (left + right) // 2
        if predicate(mid):
            result = mid
            left = mid + 1
        else:
            right = mid - 1
    return result

def solve():
    n, k = map(int, input().split())
    arr = list(map(int, input().split()))
    
    def can_achieve(x):
        # Vérifier si x est atteignable
        return True
    
    answer = binary_search_answer(can_achieve, 0, 10**9)
    print(answer)

if __name__ == "__main__":
    solve()
"""


# Snippets utiles
SNIPPETS = {
    "fast_io": """
import sys
input = sys.stdin.readline
""",
    
    "imports": """
from collections import defaultdict, deque, Counter
from heapq import heappush, heappop, heapify
from bisect import bisect_left, bisect_right
from itertools import combinations, permutations, accumulate
from functools import lru_cache
import math
""",
    
    "dfs": """
def dfs(node, graph, visited):
    visited.add(node)
    for neighbor in graph[node]:
        if neighbor not in visited:
            dfs(neighbor, graph, visited)
""",
    
    "bfs": """
from collections import deque

def bfs(start, graph):
    queue = deque([start])
    visited = {start}
    
    while queue:
        node = queue.popleft()
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    
    return visited
""",
    
    "dijkstra": """
import heapq

def dijkstra(graph, start):
    distances = {start: 0}
    pq = [(0, start)]
    
    while pq:
        current_dist, node = heapq.heappop(pq)
        
        if current_dist > distances.get(node, float('inf')):
            continue
        
        for neighbor, weight in graph[node]:
            distance = current_dist + weight
            if distance < distances.get(neighbor, float('inf')):
                distances[neighbor] = distance
                heapq.heappush(pq, (distance, neighbor))
    
    return distances
""",
    
    "prime_sieve": """
def sieve(n):
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, n + 1, i):
                is_prime[j] = False
    
    return [i for i in range(n + 1) if is_prime[i]]
""",
    
    "modular": """
MOD = 10**9 + 7

def pow_mod(base, exp, mod=MOD):
    result = 1
    base %= mod
    while exp > 0:
        if exp & 1:
            result = (result * base) % mod
        base = (base * base) % mod
        exp >>= 1
    return result

def mod_inverse(a, mod=MOD):
    return pow_mod(a, mod - 2, mod)
""",
}


def get_template(name):
    """
    Retourne un template par nom.
    
    Args:
        name: Nom du template ('basic', 'multiple', 'graph', 'dp', 'modular', 'binary_search')
        
    Returns:
        Code du template
    """
    templates = {
        'basic': TEMPLATE_BASIC,
        'multiple': TEMPLATE_MULTIPLE,
        'graph': TEMPLATE_GRAPH,
        'dp': TEMPLATE_DP,
        'modular': TEMPLATE_MODULAR,
        'binary_search': TEMPLATE_BINARY_SEARCH,
    }
    
    return templates.get(name, TEMPLATE_BASIC)


def get_snippet(name):
    """
    Retourne un snippet par nom.
    
    Args:
        name: Nom du snippet
        
    Returns:
        Code du snippet
    """
    return SNIPPETS.get(name, "")


if __name__ == "__main__":
    # Afficher tous les templates disponibles
    print("Templates disponibles:")
    print("- basic")
    print("- multiple")
    print("- graph")
    print("- dp")
    print("- modular")
    print("- binary_search")
    print()
    print("Snippets disponibles:")
    for key in SNIPPETS:
        print(f"- {key}")

