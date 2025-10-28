"""
Solutions Python pour Meta Hacker Cup (Facebook)

Exemples de solutions pour des problèmes typiques de Meta Hacker Cup.
"""

import sys
from collections import Counter, defaultdict
input = sys.stdin.readline


def travel_restrictions_2020():
    """
    Meta Hacker Cup 2020 Qualification Round - Travel Restrictions
    
    Problème: Déterminer quels pays sont accessibles depuis chaque pays
    avec des restrictions de sortie (O) et d'entrée (I).
    """
    def solve_case():
        n = int(input())
        incoming = input().strip()  # I[i] = 'Y' si on peut entrer dans i
        outgoing = input().strip()  # O[i] = 'Y' si on peut sortir de i
        
        # reachable[i][j] = peut aller de i à j
        reachable = [['N'] * n for _ in range(n)]
        
        # Depuis chaque pays de départ
        for start in range(n):
            reachable[start][start] = 'Y'
            
            # Aller à droite
            for dest in range(start + 1, n):
                if reachable[start][dest - 1] == 'Y' and \
                   outgoing[dest - 1] == 'Y' and incoming[dest] == 'Y':
                    reachable[start][dest] = 'Y'
            
            # Aller à gauche
            for dest in range(start - 1, -1, -1):
                if reachable[start][dest + 1] == 'Y' and \
                   outgoing[dest + 1] == 'Y' and incoming[dest] == 'Y':
                    reachable[start][dest] = 'Y'
        
        return [''.join(row) for row in reachable]
    
    t = int(input())
    for case_num in range(1, t + 1):
        print(f"Case #{case_num}:")
        result = solve_case()
        for row in result:
            print(row)


def alchemy_2020():
    """
    Meta Hacker Cup 2020 Qualification Round - Alchemy
    
    Problème: On a des pierres A et B. On peut combiner 2 pierres
    différentes pour en faire une du type opposé. Peut-on obtenir 1 seule pierre?
    """
    def solve_case():
        n = int(input())
        stones = input().strip()
        
        # Compter A et B
        count_a = stones.count('A')
        count_b = stones.count('B')
        
        # On peut réduire à 1 pierre ssi la différence est 1
        if abs(count_a - count_b) == 1:
            return "Y"
        else:
            return "N"
    
    t = int(input())
    for case_num in range(1, t + 1):
        answer = solve_case()
        print(f"Case #{case_num}: {answer}")


def timber_2020():
    """
    Meta Hacker Cup 2020 Round 1 - Timber
    
    Problème: Des arbres peuvent tomber à gauche ou à droite.
    Trouver la longueur maximale qu'on peut couvrir.
    """
    def solve_case():
        n = int(input())
        trees = []
        for _ in range(n):
            p, h = map(int, input().split())
            trees.append((p, h))
        
        if n == 0:
            return 0
        
        # Trier par position
        trees.sort()
        
        max_length = 0
        
        # Pour chaque arbre comme point de départ
        for start in range(n):
            # Essayer de faire tomber à gauche
            length = 0
            current_pos = trees[start][0]
            
            for i in range(start, -1, -1):
                pos, height = trees[i]
                if current_pos <= pos + height:
                    length += max(0, min(current_pos, pos + height) - pos)
                    current_pos = pos
                else:
                    break
            
            max_length = max(max_length, length)
            
            # Essayer de faire tomber à droite
            length = 0
            current_pos = trees[start][0]
            
            for i in range(start, n):
                pos, height = trees[i]
                if current_pos >= pos - height:
                    length += max(0, pos + height - max(current_pos, pos))
                    current_pos = pos + height
                else:
                    break
            
            max_length = max(max_length, length)
        
        return max_length
    
    t = int(input())
    for case_num in range(1, t + 1):
        answer = solve_case()
        print(f"Case #{case_num}: {answer}")


def consistency_2019():
    """
    Meta Hacker Cup 2019 Qualification Round - Consistency (Chapter 1)
    
    Problème: Transformer une chaîne pour que tous les caractères soient
    dans la même classe (voyelles ou consonnes). Coût = 1 par transformation.
    """
    def solve_case():
        s = input().strip()
        vowels = set('AEIOU')
        
        vowel_count = sum(1 for c in s if c in vowels)
        consonant_count = len(s) - vowel_count
        
        # Option 1: Tout en voyelles
        cost1 = consonant_count
        
        # Option 2: Tout en consonnes
        cost2 = vowel_count
        
        return min(cost1, cost2)
    
    t = int(input())
    for case_num in range(1, t + 1):
        answer = solve_case()
        print(f"Case #{case_num}: {answer}")


def running_on_fumes_2022():
    """
    Meta Hacker Cup 2022 Practice Round - Running on Fumes (Chapter 1)
    
    Problème: Voyager de la ville 1 à N en voiture. On doit faire le plein
    dans des stations (coûts différents). La voiture peut parcourir M km.
    """
    def solve_case():
        n, m = map(int, input().split())
        costs = [0] + list(map(int, input().split()))  # 1-indexed
        
        INF = float('inf')
        # dp[i] = coût minimum pour atteindre la ville i
        dp = [INF] * (n + 1)
        dp[1] = costs[1]
        
        for i in range(2, n + 1):
            # Essayer de venir de toutes les villes dans la portée
            for j in range(max(1, i - m), i):
                if dp[j] != INF and costs[j] > 0:
                    dp[i] = min(dp[i], dp[j] + costs[i])
        
        # Le dernier plein n'est pas nécessaire si on arrive
        result = INF
        for j in range(max(1, n - m + 1), n + 1):
            if dp[j] != INF:
                result = min(result, dp[j] if j == n and costs[n] == 0 else dp[j])
        
        return result if result != INF else -1
    
    t = int(input())
    for case_num in range(1, t + 1):
        answer = solve_case()
        print(f"Case #{case_num}: {answer}")


def second_hands_2022():
    """
    Meta Hacker Cup 2022 Qualification Round - Second Hands
    
    Problème: Distribuer N objets dans 2 étagères de capacité K.
    Chaque étagère ne peut avoir qu'un exemplaire de chaque type.
    """
    def solve_case():
        n, k = map(int, input().split())
        styles = list(map(int, input().split()))
        
        # Compter les occurrences
        count = Counter(styles)
        
        # Impossible si:
        # 1. Plus de 2 exemplaires d'un même type
        # 2. Total trop grand pour 2 étagères
        if any(c > 2 for c in count.values()):
            return "NO"
        
        if n > 2 * k:
            return "NO"
        
        # Répartir les objets
        shelf1 = []
        shelf2 = []
        
        for style, cnt in count.items():
            if cnt == 2:
                shelf1.append(style)
                shelf2.append(style)
            elif len(shelf1) < k:
                shelf1.append(style)
            else:
                shelf2.append(style)
        
        if len(shelf1) <= k and len(shelf2) <= k:
            return "YES"
        else:
            return "NO"
    
    t = int(input())
    for case_num in range(1, t + 1):
        answer = solve_case()
        print(f"Case #{case_num}: {answer}")


# Template pour exécution
if __name__ == "__main__":
    # Décommenter la fonction à tester
    # travel_restrictions_2020()
    # alchemy_2020()
    # consistency_2019()
    # second_hands_2022()
    pass

