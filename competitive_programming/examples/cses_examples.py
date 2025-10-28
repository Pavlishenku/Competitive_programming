"""
Solutions Python pour CSES Problem Set

Exemples de solutions pour des problèmes du CSES Problem Set.
Site: https://cses.fi/problemset/
"""

import sys
from collections import deque, defaultdict
input = sys.stdin.readline


def weird_algorithm():
    """
    CSES - Introductory Problems - Weird Algorithm
    
    Problème: Simuler la séquence de Collatz.
    """
    def solve():
        n = int(input())
        result = [n]
        
        while n != 1:
            if n % 2 == 0:
                n //= 2
            else:
                n = n * 3 + 1
            result.append(n)
        
        print(' '.join(map(str, result)))
    
    solve()


def missing_number():
    """
    CSES - Introductory Problems - Missing Number
    
    Problème: Trouver le nombre manquant de 1 à n.
    """
    def solve():
        n = int(input())
        numbers = list(map(int, input().split()))
        
        # Somme attendue - somme réelle
        expected = n * (n + 1) // 2
        actual = sum(numbers)
        
        print(expected - actual)
    
    solve()


def repetitions():
    """
    CSES - Introductory Problems - Repetitions
    
    Problème: Plus longue séquence de caractères identiques.
    """
    def solve():
        s = input().strip()
        
        max_len = 1
        current_len = 1
        
        for i in range(1, len(s)):
            if s[i] == s[i-1]:
                current_len += 1
                max_len = max(max_len, current_len)
            else:
                current_len = 1
        
        print(max_len)
    
    solve()


def ferris_wheel():
    """
    CSES - Sorting and Searching - Ferris Wheel
    
    Problème: Minimiser le nombre de gondoles.
    """
    def solve():
        n, x = map(int, input().split())
        weights = list(map(int, input().split()))
        
        weights.sort()
        
        left = 0
        right = n - 1
        gondolas = 0
        
        while left <= right:
            if weights[left] + weights[right] <= x:
                left += 1
            right -= 1
            gondolas += 1
        
        print(gondolas)
    
    solve()


def distinct_numbers():
    """
    CSES - Sorting and Searching - Distinct Numbers
    
    Problème: Compter les nombres distincts.
    """
    def solve():
        n = int(input())
        numbers = list(map(int, input().split()))
        
        print(len(set(numbers)))
    
    solve()


def concert_tickets():
    """
    CSES - Sorting and Searching - Concert Tickets
    
    Problème: Assigner des tickets aux clients (greedy).
    """
    def solve():
        n, m = map(int, input().split())
        prices = list(map(int, input().split()))
        max_prices = list(map(int, input().split()))
        
        # Utiliser multiset (simulé avec dict)
        from collections import Counter
        available = Counter(prices)
        
        for max_price in max_prices:
            # Trouver le plus grand prix <= max_price
            found = False
            for price in sorted(available.keys(), reverse=True):
                if price <= max_price and available[price] > 0:
                    print(price)
                    available[price] -= 1
                    if available[price] == 0:
                        del available[price]
                    found = True
                    break
            
            if not found:
                print(-1)
    
    solve()


def sum_of_two_values():
    """
    CSES - Sorting and Searching - Sum of Two Values
    
    Problème: Two sum problem.
    """
    def solve():
        n, x = map(int, input().split())
        values = list(map(int, input().split()))
        
        seen = {}
        
        for i, val in enumerate(values):
            complement = x - val
            if complement in seen:
                print(seen[complement] + 1, i + 1)
                return
            seen[val] = i
        
        print("IMPOSSIBLE")
    
    solve()


def dice_combinations():
    """
    CSES - Dynamic Programming - Dice Combinations
    
    Problème: Nombre de façons d'obtenir une somme avec un dé.
    """
    def solve():
        n = int(input())
        MOD = 10**9 + 7
        
        dp = [0] * (n + 1)
        dp[0] = 1
        
        for i in range(1, n + 1):
            for dice in range(1, 7):
                if i >= dice:
                    dp[i] = (dp[i] + dp[i - dice]) % MOD
        
        print(dp[n])
    
    solve()


def minimizing_coins():
    """
    CSES - Dynamic Programming - Minimizing Coins
    
    Problème: Coin change minimum.
    """
    def solve():
        n, x = map(int, input().split())
        coins = list(map(int, input().split()))
        
        dp = [float('inf')] * (x + 1)
        dp[0] = 0
        
        for i in range(1, x + 1):
            for coin in coins:
                if i >= coin:
                    dp[i] = min(dp[i], dp[i - coin] + 1)
        
        if dp[x] == float('inf'):
            print(-1)
        else:
            print(dp[x])
    
    solve()


def coin_combinations_1():
    """
    CSES - Dynamic Programming - Coin Combinations I
    
    Problème: Nombre de façons de faire une somme (ordre compte).
    """
    def solve():
        n, x = map(int, input().split())
        coins = list(map(int, input().split()))
        
        MOD = 10**9 + 7
        dp = [0] * (x + 1)
        dp[0] = 1
        
        for i in range(1, x + 1):
            for coin in coins:
                if i >= coin:
                    dp[i] = (dp[i] + dp[i - coin]) % MOD
        
        print(dp[x])
    
    solve()


def shortest_routes_1():
    """
    CSES - Graph Algorithms - Shortest Routes I
    
    Problème: Dijkstra depuis un sommet.
    """
    def solve():
        import heapq
        
        n, m = map(int, input().split())
        graph = [[] for _ in range(n + 1)]
        
        for _ in range(m):
            a, b, c = map(int, input().split())
            graph[a].append((b, c))
        
        # Dijkstra
        dist = [float('inf')] * (n + 1)
        dist[1] = 0
        pq = [(0, 1)]
        
        while pq:
            d, u = heapq.heappop(pq)
            
            if d > dist[u]:
                continue
            
            for v, w in graph[u]:
                if dist[u] + w < dist[v]:
                    dist[v] = dist[u] + w
                    heapq.heappush(pq, (dist[v], v))
        
        print(' '.join(map(str, dist[1:n+1])))
    
    solve()


def building_roads():
    """
    CSES - Graph Algorithms - Building Roads
    
    Problème: Connecter toutes les composantes connexes.
    """
    def solve():
        n, m = map(int, input().split())
        
        # DSU
        parent = list(range(n + 1))
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
                return True
            return False
        
        for _ in range(m):
            a, b = map(int, input().split())
            union(a, b)
        
        # Trouver les représentants
        components = set()
        for i in range(1, n + 1):
            components.add(find(i))
        
        components = sorted(components)
        
        print(len(components) - 1)
        for i in range(len(components) - 1):
            print(components[i], components[i + 1])
    
    solve()


def message_route():
    """
    CSES - Graph Algorithms - Message Route
    
    Problème: Plus court chemin avec BFS.
    """
    def solve():
        n, m = map(int, input().split())
        graph = [[] for _ in range(n + 1)]
        
        for _ in range(m):
            a, b = map(int, input().split())
            graph[a].append(b)
            graph[b].append(a)
        
        # BFS
        from collections import deque
        queue = deque([1])
        visited = {1}
        parent = [-1] * (n + 1)
        
        while queue:
            u = queue.popleft()
            
            if u == n:
                break
            
            for v in graph[u]:
                if v not in visited:
                    visited.add(v)
                    parent[v] = u
                    queue.append(v)
        
        if parent[n] == -1:
            print("IMPOSSIBLE")
        else:
            path = []
            current = n
            while current != -1:
                path.append(current)
                current = parent[current]
            
            path.reverse()
            print(len(path))
            print(' '.join(map(str, path)))
    
    solve()


# Template pour exécution
if __name__ == "__main__":
    # Décommenter la fonction à tester
    # weird_algorithm()
    # missing_number()
    # dice_combinations()
    pass

