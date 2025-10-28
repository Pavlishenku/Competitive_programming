"""
Solutions Python pour Google Kick Start

Exemples de solutions pour des problèmes typiques de Google Kick Start.
"""

import sys
from collections import deque, defaultdict
input = sys.stdin.readline


def allocation_2020():
    """
    Kick Start 2020 Round A - Allocation
    
    Problème: Acheter le maximum de maisons avec un budget B.
    """
    def solve_case():
        n, b = map(int, input().split())
        prices = list(map(int, input().split()))
        
        # Trier et acheter les moins chères
        prices.sort()
        
        count = 0
        total = 0
        
        for price in prices:
            if total + price <= b:
                total += price
                count += 1
            else:
                break
        
        return count
    
    t = int(input())
    for case_num in range(1, t + 1):
        answer = solve_case()
        print(f"Case #{case_num}: {answer}")


def plates_2020():
    """
    Kick Start 2020 Round A - Plates
    
    Problème: Choisir P assiettes de N piles de K assiettes pour
    maximiser la beauté totale. On ne peut prendre que du dessus.
    """
    def solve_case():
        n, k, p = map(int, input().split())
        stacks = []
        
        for _ in range(n):
            plates = list(map(int, input().split()))
            # Précalculer les sommes de préfixes
            prefix = [0]
            for plate in plates:
                prefix.append(prefix[-1] + plate)
            stacks.append(prefix)
        
        # dp[i][j] = beauté max en prenant j assiettes des i premières piles
        dp = [[-1] * (p + 1) for _ in range(n + 1)]
        dp[0][0] = 0
        
        for i in range(1, n + 1):
            for j in range(p + 1):
                # Essayer de prendre t assiettes de la pile i
                for t in range(min(k, j) + 1):
                    if dp[i-1][j-t] != -1:
                        beauty = dp[i-1][j-t] + stacks[i-1][t]
                        dp[i][j] = max(dp[i][j], beauty)
        
        return dp[n][p]
    
    t = int(input())
    for case_num in range(1, t + 1):
        answer = solve_case()
        print(f"Case #{case_num}: {answer}")


def workout_2020():
    """
    Kick Start 2020 Round A - Workout
    
    Problème: Ajouter au plus K sessions d'entraînement pour minimiser
    la différence maximale entre sessions consécutives.
    """
    def can_achieve(minutes, k, max_diff):
        """Vérifie si on peut avoir max_diff en ajoutant k sessions"""
        needed = 0
        
        for i in range(len(minutes) - 1):
            diff = minutes[i+1] - minutes[i]
            if diff > max_diff:
                # Nombre de sessions à ajouter
                needed += (diff - 1) // max_diff
        
        return needed <= k
    
    def solve_case():
        n, k = map(int, input().split())
        minutes = list(map(int, input().split()))
        
        # Binary search sur la réponse
        left = 1
        right = max(minutes[i+1] - minutes[i] for i in range(n-1))
        
        result = right
        
        while left <= right:
            mid = (left + right) // 2
            
            if can_achieve(minutes, k, mid):
                result = mid
                right = mid - 1
            else:
                left = mid + 1
        
        return result
    
    t = int(input())
    for case_num in range(1, t + 1):
        answer = solve_case()
        print(f"Case #{case_num}: {answer}")


def bus_routes_2020():
    """
    Kick Start 2020 Round B - Bus Routes
    
    Problème: Prendre N bus avec des intervalles Xi. Dernier départ <= D.
    Trouver le plus tard qu'on peut partir.
    """
    def solve_case():
        n, d = map(int, input().split())
        intervals = list(map(int, input().split()))
        
        # Partir au plus tard pour chaque bus
        current_time = d
        
        for i in range(n - 1, -1, -1):
            x = intervals[i]
            # Le dernier multiple de x <= current_time
            current_time = (current_time // x) * x
        
        return current_time
    
    t = int(input())
    for case_num in range(1, t + 1):
        answer = solve_case()
        print(f"Case #{case_num}: {answer}")


def robot_path_decoding_2020():
    """
    Kick Start 2020 Round B - Robot Path Decoding
    
    Problème: Décoder un chemin de robot avec répétitions (format compressé).
    Exemple: "2(3(NE))" = "NENENENE NENENENE"
    """
    def solve_case():
        program = input().strip()
        MOD = 10**9
        
        def decode(s, start_idx):
            """Retourne (end_row, end_col, next_idx)"""
            row, col = 0, 0
            i = start_idx
            
            while i < len(s) and s[i] != ')':
                if s[i] == 'N':
                    row = (row - 1) % MOD
                    i += 1
                elif s[i] == 'S':
                    row = (row + 1) % MOD
                    i += 1
                elif s[i] == 'E':
                    col = (col + 1) % MOD
                    i += 1
                elif s[i] == 'W':
                    col = (col - 1) % MOD
                    i += 1
                elif s[i].isdigit():
                    mult = int(s[i])
                    i += 2  # Skip digit and '('
                    sub_row, sub_col, i = decode(s, i)
                    i += 1  # Skip ')'
                    row = (row + mult * sub_row) % MOD
                    col = (col + mult * sub_col) % MOD
            
            return row, col, i
        
        row, col, _ = decode(program, 0)
        
        # Convert to 1-indexed
        row = (row % MOD) + 1
        col = (col % MOD) + 1
        
        return f"{col} {row}"
    
    t = int(input())
    for case_num in range(1, t + 1):
        answer = solve_case()
        print(f"Case #{case_num}: {answer}")


def combination_lock_2021():
    """
    Kick Start 2021 Round A - L Shaped Plots
    
    Problème: Compter les parcelles en forme de L dans une grille.
    """
    def solve_case():
        r, c = map(int, input().split())
        grid = []
        for _ in range(r):
            row = list(map(int, input().split()))
            grid.append(row)
        
        # Pour chaque cellule, compter les 1s consécutifs dans 4 directions
        up = [[0] * c for _ in range(r)]
        down = [[0] * c for _ in range(r)]
        left = [[0] * c for _ in range(r)]
        right = [[0] * c for _ in range(r)]
        
        for i in range(r):
            for j in range(c):
                if grid[i][j] == 1:
                    up[i][j] = 1 if i == 0 else up[i-1][j] + 1
                    left[i][j] = 1 if j == 0 else left[i][j-1] + 1
        
        for i in range(r-1, -1, -1):
            for j in range(c-1, -1, -1):
                if grid[i][j] == 1:
                    down[i][j] = 1 if i == r-1 else down[i+1][j] + 1
                    right[i][j] = 1 if j == c-1 else right[i][j+1] + 1
        
        count = 0
        
        # Pour chaque cellule, vérifier les 8 configurations de L
        for i in range(r):
            for j in range(c):
                if grid[i][j] == 1:
                    # L vers le haut-droite
                    for k in range(2, min(up[i][j], right[i][j] // 2) + 1):
                        count += 1
                    for k in range(2, min(up[i][j] // 2, right[i][j]) + 1):
                        count += 1
                    
                    # L vers le haut-gauche
                    for k in range(2, min(up[i][j], left[i][j] // 2) + 1):
                        count += 1
                    for k in range(2, min(up[i][j] // 2, left[i][j]) + 1):
                        count += 1
                    
                    # L vers le bas-droite
                    for k in range(2, min(down[i][j], right[i][j] // 2) + 1):
                        count += 1
                    for k in range(2, min(down[i][j] // 2, right[i][j]) + 1):
                        count += 1
                    
                    # L vers le bas-gauche
                    for k in range(2, min(down[i][j], left[i][j] // 2) + 1):
                        count += 1
                    for k in range(2, min(down[i][j] // 2, left[i][j]) + 1):
                        count += 1
        
        return count
    
    t = int(input())
    for case_num in range(1, t + 1):
        answer = solve_case()
        print(f"Case #{case_num}: {answer}")


# Template pour exécution
if __name__ == "__main__":
    # Décommenter la fonction à tester
    # allocation_2020()
    # plates_2020()
    # workout_2020()
    # bus_routes_2020()
    pass

