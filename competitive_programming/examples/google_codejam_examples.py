"""
Solutions Python pour Google Code Jam

Exemples de solutions pour des problèmes typiques de Google Code Jam.
"""

import sys
input = sys.stdin.readline


def vestigium_2020():
    """
    Google Code Jam 2020 Qualification Round - Vestigium
    
    Problème: Analyser une matrice N×N pour trouver:
    - La trace (somme diagonale)
    - Nombre de lignes avec doublons
    - Nombre de colonnes avec doublons
    
    Input:
        T cas de test
        Pour chaque cas: N (taille matrice), puis N lignes de N entiers
    """
    def solve_case():
        n = int(input())
        matrix = []
        for _ in range(n):
            row = list(map(int, input().split()))
            matrix.append(row)
        
        # Calculer la trace
        trace = sum(matrix[i][i] for i in range(n))
        
        # Compter lignes avec doublons
        rows_duplicates = sum(1 for row in matrix if len(set(row)) < n)
        
        # Compter colonnes avec doublons
        cols_duplicates = 0
        for j in range(n):
            col = [matrix[i][j] for i in range(n)]
            if len(set(col)) < n:
                cols_duplicates += 1
        
        return trace, rows_duplicates, cols_duplicates
    
    t = int(input())
    for case_num in range(1, t + 1):
        trace, rows, cols = solve_case()
        print(f"Case #{case_num}: {trace} {rows} {cols}")


def nesting_depth_2020():
    """
    Google Code Jam 2020 Qualification Round - Nesting Depth
    
    Problème: Ajouter des parenthèses à une chaîne de chiffres pour que
    chaque chiffre d ait une profondeur d'imbrication de d.
    
    Example: "312" -> "((3)1(2))"
    """
    def solve_case():
        s = input().strip()
        result = []
        depth = 0
        
        for char in s:
            target = int(char)
            
            # Ajouter ou retirer des parenthèses
            if target > depth:
                result.append('(' * (target - depth))
            elif target < depth:
                result.append(')' * (depth - target))
            
            result.append(char)
            depth = target
        
        # Fermer toutes les parenthèses restantes
        result.append(')' * depth)
        
        return ''.join(result)
    
    t = int(input())
    for case_num in range(1, t + 1):
        answer = solve_case()
        print(f"Case #{case_num}: {answer}")


def parenting_partnering_2020():
    """
    Google Code Jam 2020 Qualification Round - Parenting Partnering Returns
    
    Problème: Assigner des activités à C et J sans chevauchement.
    
    Input: N activités avec (start, end)
    Output: String de C et J, ou "IMPOSSIBLE"
    """
    def solve_case():
        n = int(input())
        activities = []
        for i in range(n):
            start, end = map(int, input().split())
            activities.append((start, end, i))
        
        # Trier par heure de début
        activities.sort()
        
        result = [''] * n
        c_free = 0
        j_free = 0
        
        for start, end, idx in activities:
            if c_free <= start:
                result[idx] = 'C'
                c_free = end
            elif j_free <= start:
                result[idx] = 'J'
                j_free = end
            else:
                return "IMPOSSIBLE"
        
        return ''.join(result)
    
    t = int(input())
    for case_num in range(1, t + 1):
        answer = solve_case()
        print(f"Case #{case_num}: {answer}")


def indicium_2020():
    """
    Google Code Jam 2020 Qualification Round - Indicium
    
    Problème: Construire un carré latin N×N avec trace K.
    
    Utilise backtracking pour construire le carré.
    """
    def is_valid(board, row, col, num, n):
        # Vérifier ligne
        if num in board[row]:
            return False
        
        # Vérifier colonne
        if any(board[i][col] == num for i in range(n)):
            return False
        
        return True
    
    def solve_latin(board, row, col, n, target_trace, current_trace):
        if row == n:
            return current_trace == target_trace
        
        next_row = row if col < n - 1 else row + 1
        next_col = (col + 1) % n
        
        for num in range(1, n + 1):
            if is_valid(board, row, col, num, n):
                board[row][col] = num
                
                new_trace = current_trace
                if row == col:
                    new_trace += num
                
                if solve_latin(board, next_row, next_col, n, target_trace, new_trace):
                    return True
                
                board[row][col] = 0
        
        return False
    
    def solve_case():
        n, k = map(int, input().split())
        
        # Vérification rapide
        if k < n or k > n * n:
            return "IMPOSSIBLE", None
        
        board = [[0] * n for _ in range(n)]
        
        if solve_latin(board, 0, 0, n, k, 0):
            return "POSSIBLE", board
        else:
            return "IMPOSSIBLE", None
    
    t = int(input())
    for case_num in range(1, t + 1):
        status, board = solve_case()
        print(f"Case #{case_num}: {status}")
        if board:
            for row in board:
                print(' '.join(map(str, row)))


def reversort_2021():
    """
    Google Code Jam 2021 Qualification Round - Reversort
    
    Problème: Simuler l'algorithme Reversort et compter les opérations.
    """
    def reversort(arr):
        cost = 0
        n = len(arr)
        
        for i in range(n - 1):
            # Trouver le minimum dans arr[i:]
            min_idx = i + arr[i:].index(min(arr[i:]))
            
            # Reverser arr[i:min_idx+1]
            arr[i:min_idx+1] = arr[i:min_idx+1][::-1]
            
            # Ajouter le coût
            cost += min_idx - i + 1
        
        return cost
    
    t = int(input())
    for case_num in range(1, t + 1):
        n = int(input())
        arr = list(map(int, input().split()))
        cost = reversort(arr)
        print(f"Case #{case_num}: {cost}")


def moons_and_umbrellas_2021():
    """
    Google Code Jam 2021 Qualification Round - Moons and Umbrellas
    
    Problème: Remplir une chaîne avec C et J pour minimiser le coût.
    CJ coûte X, JC coûte Y.
    """
    def solve_case():
        line = input().split()
        x, y = int(line[0]), int(line[1])
        s = line[2]
        
        n = len(s)
        # dp[i][c] = coût minimum pour remplir s[:i] avec dernier caractère c
        INF = float('inf')
        dp = [[INF, INF] for _ in range(n + 1)]  # [C, J]
        dp[0][0] = dp[0][1] = 0
        
        for i in range(n):
            if s[i] == 'C' or s[i] == '?':
                dp[i+1][0] = min(dp[i+1][0], dp[i][0])
                dp[i+1][0] = min(dp[i+1][0], dp[i][1] + y)
            
            if s[i] == 'J' or s[i] == '?':
                dp[i+1][1] = min(dp[i+1][1], dp[i][1])
                dp[i+1][1] = min(dp[i+1][1], dp[i][0] + x)
        
        return min(dp[n])
    
    t = int(input())
    for case_num in range(1, t + 1):
        cost = solve_case()
        print(f"Case #{case_num}: {cost}")


# Template pour exécution
if __name__ == "__main__":
    # Décommenter la fonction à tester
    # vestigium_2020()
    # nesting_depth_2020()
    # parenting_partnering_2020()
    # reversort_2021()
    # moons_and_umbrellas_2021()
    pass

