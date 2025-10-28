"""
Solutions Python pour AtCoder

Exemples de solutions pour des problèmes typiques d'AtCoder.
"""

import sys
from collections import deque, defaultdict
input = sys.stdin.readline


def frog_1_abc():
    """
    AtCoder Beginner Contest - DP A - Frog 1
    
    Problème: Grenouille peut sauter de 1 ou 2 pierres.
    Minimiser le coût total.
    """
    def solve():
        n = int(input())
        h = list(map(int, input().split()))
        
        dp = [float('inf')] * n
        dp[0] = 0
        
        for i in range(1, n):
            # Sauter de i-1
            dp[i] = min(dp[i], dp[i-1] + abs(h[i] - h[i-1]))
            
            # Sauter de i-2
            if i >= 2:
                dp[i] = min(dp[i], dp[i-2] + abs(h[i] - h[i-2]))
        
        print(dp[n-1])
    
    solve()


def knapsack_dp():
    """
    AtCoder DP Contest - D - Knapsack 1
    
    Problème: Knapsack classique.
    """
    def solve():
        n, w = map(int, input().split())
        items = []
        for _ in range(n):
            weight, value = map(int, input().split())
            items.append((weight, value))
        
        dp = [0] * (w + 1)
        
        for weight, value in items:
            for capacity in range(w, weight - 1, -1):
                dp[capacity] = max(dp[capacity], dp[capacity - weight] + value)
        
        print(dp[w])
    
    solve()


def lcs_dp():
    """
    AtCoder DP Contest - F - LCS
    
    Problème: Plus longue sous-séquence commune.
    """
    def solve():
        s = input().strip()
        t = input().strip()
        
        m, n = len(s), len(t)
        dp = [[""] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s[i-1] == t[j-1]:
                    dp[i][j] = dp[i-1][j-1] + s[i-1]
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1], key=len)
        
        print(dp[m][n])
    
    solve()


def grid_1_dp():
    """
    AtCoder DP Contest - H - Grid 1
    
    Problème: Nombre de chemins dans une grille avec obstacles.
    """
    def solve():
        h, w = map(int, input().split())
        grid = []
        for _ in range(h):
            row = input().strip()
            grid.append(row)
        
        MOD = 10**9 + 7
        dp = [[0] * w for _ in range(h)]
        
        # Initialiser
        dp[0][0] = 1 if grid[0][0] == '.' else 0
        
        for i in range(h):
            for j in range(w):
                if grid[i][j] == '#':
                    continue
                
                if i > 0:
                    dp[i][j] = (dp[i][j] + dp[i-1][j]) % MOD
                if j > 0:
                    dp[i][j] = (dp[i][j] + dp[i][j-1]) % MOD
        
        print(dp[h-1][w-1])
    
    solve()


def independent_set_dp():
    """
    AtCoder DP Contest - P - Independent Set
    
    Problème: Compter les ensembles indépendants dans un arbre.
    """
    def solve():
        n = int(input())
        tree = [[] for _ in range(n)]
        
        for _ in range(n - 1):
            u, v = map(int, input().split())
            u -= 1
            v -= 1
            tree[u].append(v)
            tree[v].append(u)
        
        MOD = 10**9 + 7
        
        # dp[v][0] = nombre de façons si v n'est pas pris
        # dp[v][1] = nombre de façons si v est pris
        dp = [[1, 1] for _ in range(n)]
        
        def dfs(v, parent):
            for u in tree[v]:
                if u == parent:
                    continue
                
                dfs(u, v)
                
                # Si v n'est pas pris, u peut être pris ou non
                dp[v][0] = (dp[v][0] * (dp[u][0] + dp[u][1])) % MOD
                
                # Si v est pris, u ne peut pas être pris
                dp[v][1] = (dp[v][1] * dp[u][0]) % MOD
        
        dfs(0, -1)
        print((dp[0][0] + dp[0][1]) % MOD)
    
    solve()


def xor_matching():
    """
    AtCoder ABC141 E - Who Says a Pun?
    
    Problème: Trouver la plus longue sous-chaîne apparaissant deux fois
    sans chevauchement.
    """
    def solve():
        n = int(input())
        s = input().strip()
        
        # Binary search sur la longueur
        def check(length):
            seen = {}
            for i in range(n - length + 1):
                substring = s[i:i+length]
                if substring in seen:
                    if i >= seen[substring] + length:
                        return True
                else:
                    seen[substring] = i
            return False
        
        left, right = 0, n // 2
        answer = 0
        
        while left <= right:
            mid = (left + right) // 2
            if check(mid):
                answer = mid
                left = mid + 1
            else:
                right = mid - 1
        
        print(answer)
    
    solve()


def string_formation():
    """
    AtCoder ABC158 E - Divisible Substring
    
    Problème: Compter les sous-chaînes dont la valeur numérique
    est divisible par P.
    """
    def solve():
        n, p = map(int, input().split())
        s = input().strip()
        
        if p == 2 or p == 5:
            # Cas spécial
            count = 0
            for i in range(n):
                if int(s[i]) % p == 0:
                    count += i + 1
            print(count)
            return
        
        # Utiliser les restes
        from collections import defaultdict
        remainder_count = defaultdict(int)
        remainder_count[0] = 1
        
        current_remainder = 0
        power = 1
        count = 0
        
        for i in range(n - 1, -1, -1):
            digit = int(s[i])
            current_remainder = (current_remainder + digit * power) % p
            count += remainder_count[current_remainder]
            remainder_count[current_remainder] += 1
            power = (power * 10) % p
        
        print(count)
    
    solve()


def colorful_path():
    """
    AtCoder ABC146 D - Coloring Edges on Tree
    
    Problème: Colorer les arêtes d'un arbre tel que deux arêtes
    adjacentes aient des couleurs différentes.
    """
    def solve():
        n = int(input())
        tree = [[] for _ in range(n)]
        edges = []
        
        for i in range(n - 1):
            a, b = map(int, input().split())
            a -= 1
            b -= 1
            tree[a].append((b, i))
            tree[b].append((a, i))
            edges.append((a, b))
        
        color = [0] * (n - 1)
        max_colors = 0
        
        def dfs(v, parent, parent_color):
            nonlocal max_colors
            current_color = 1
            
            for u, edge_idx in tree[v]:
                if u == parent:
                    continue
                
                if current_color == parent_color:
                    current_color += 1
                
                color[edge_idx] = current_color
                max_colors = max(max_colors, current_color)
                
                dfs(u, v, current_color)
                current_color += 1
        
        dfs(0, -1, 0)
        
        print(max_colors)
        for c in color:
            print(c)
    
    solve()


# Template pour exécution
if __name__ == "__main__":
    # Décommenter la fonction à tester
    # frog_1_abc()
    # knapsack_dp()
    # lcs_dp()
    pass

