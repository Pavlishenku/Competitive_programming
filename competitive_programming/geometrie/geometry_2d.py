"""
Géométrie 2D (Points, Vecteurs, Segments)

Description:
    Classes et fonctions pour géométrie computationnelle 2D:
    - Points et vecteurs
    - Produits scalaire et vectoriel
    - Intersections de segments
    - Distances et projections

Complexité:
    - Opérations de base: O(1)
    - Convex Hull: O(n log n)

Cas d'usage:
    - Problèmes géométriques
    - Détection de collisions
    - Calculs de distances
    - Point in polygon
    
Problèmes types:
    - Codeforces: 166B, 166C, 357C
    - AtCoder: ABC139F, ABC151F
    - CSES: Point Location Test
    
Implémentation par: 2025-10-27
Testé: Oui
"""

import math


class Point:
    """Point 2D avec opérations vectorielles."""
    
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y)
    
    def __sub__(self, other):
        return Point(self.x - other.x, self.y - other.y)
    
    def __mul__(self, scalar):
        return Point(self.x * scalar, self.y * scalar)
    
    def __truediv__(self, scalar):
        return Point(self.x / scalar, self.y / scalar)
    
    def __eq__(self, other):
        return abs(self.x - other.x) < 1e-9 and abs(self.y - other.y) < 1e-9
    
    def __repr__(self):
        return f"Point({self.x}, {self.y})"
    
    def __hash__(self):
        return hash((self.x, self.y))
    
    def distance_to(self, other):
        """Distance euclidienne à un autre point."""
        dx = self.x - other.x
        dy = self.y - other.y
        return math.sqrt(dx*dx + dy*dy)
    
    def distance_squared(self, other):
        """Distance au carré (évite sqrt)."""
        dx = self.x - other.x
        dy = self.y - other.y
        return dx*dx + dy*dy
    
    def dot(self, other):
        """Produit scalaire."""
        return self.x * other.x + self.y * other.y
    
    def cross(self, other):
        """Produit vectoriel (en 2D, retourne un scalaire)."""
        return self.x * other.y - self.y * other.x
    
    def length(self):
        """Norme du vecteur."""
        return math.sqrt(self.x*self.x + self.y*self.y)
    
    def normalize(self):
        """Retourne le vecteur normalisé."""
        l = self.length()
        if l == 0:
            return Point(0, 0)
        return Point(self.x / l, self.y / l)
    
    def rotate(self, angle):
        """Rotation de angle radians autour de l'origine."""
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        return Point(
            self.x * cos_a - self.y * sin_a,
            self.x * sin_a + self.y * cos_a
        )


def ccw(a, b, c):
    """
    Test d'orientation: sens trigonométrique (counter-clockwise).
    
    Args:
        a, b, c: Points
        
    Returns:
        > 0 si sens trigo, < 0 si horaire, = 0 si alignés
    """
    return (b - a).cross(c - a)


def segments_intersect(p1, p2, p3, p4):
    """
    Vérifie si les segments [p1, p2] et [p3, p4] s'intersectent.
    
    Args:
        p1, p2: Extrémités du premier segment
        p3, p4: Extrémités du deuxième segment
        
    Returns:
        True si les segments s'intersectent
    """
    d1 = ccw(p3, p4, p1)
    d2 = ccw(p3, p4, p2)
    d3 = ccw(p1, p2, p3)
    d4 = ccw(p1, p2, p4)
    
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    
    # Cas colinéaires
    if d1 == 0 and on_segment(p3, p1, p4):
        return True
    if d2 == 0 and on_segment(p3, p2, p4):
        return True
    if d3 == 0 and on_segment(p1, p3, p2):
        return True
    if d4 == 0 and on_segment(p1, p4, p2):
        return True
    
    return False


def on_segment(p, q, r):
    """Vérifie si q est sur le segment [p, r] (points alignés)."""
    return (min(p.x, r.x) <= q.x <= max(p.x, r.x) and
            min(p.y, r.y) <= q.y <= max(p.y, r.y))


def point_to_line_distance(point, line_p1, line_p2):
    """
    Distance d'un point à une droite.
    
    Args:
        point: Point
        line_p1, line_p2: Deux points définissant la droite
        
    Returns:
        Distance
    """
    numerator = abs(ccw(line_p1, line_p2, point))
    denominator = line_p1.distance_to(line_p2)
    
    if denominator == 0:
        return point.distance_to(line_p1)
    
    return numerator / denominator


def point_to_segment_distance(point, seg_p1, seg_p2):
    """
    Distance d'un point à un segment.
    
    Args:
        point: Point
        seg_p1, seg_p2: Extrémités du segment
        
    Returns:
        Distance minimale
    """
    v = seg_p2 - seg_p1
    w = point - seg_p1
    
    c1 = w.dot(v)
    if c1 <= 0:
        return point.distance_to(seg_p1)
    
    c2 = v.dot(v)
    if c1 >= c2:
        return point.distance_to(seg_p2)
    
    b = c1 / c2
    pb = seg_p1 + v * b
    return point.distance_to(pb)


def polygon_area(points):
    """
    Calcule l'aire d'un polygone (Shoelace formula).
    
    Args:
        points: Liste de Points (dans l'ordre)
        
    Returns:
        Aire du polygone
    """
    n = len(points)
    area = 0
    
    for i in range(n):
        j = (i + 1) % n
        area += points[i].x * points[j].y
        area -= points[j].x * points[i].y
    
    return abs(area) / 2


def point_in_polygon(point, polygon):
    """
    Vérifie si un point est à l'intérieur d'un polygone (ray casting).
    
    Args:
        point: Point à tester
        polygon: Liste de Points formant le polygone
        
    Returns:
        True si le point est à l'intérieur
    """
    n = len(polygon)
    inside = False
    
    p1 = polygon[0]
    for i in range(1, n + 1):
        p2 = polygon[i % n]
        
        if point.y > min(p1.y, p2.y):
            if point.y <= max(p1.y, p2.y):
                if point.x <= max(p1.x, p2.x):
                    if p1.y != p2.y:
                        xinters = (point.y - p1.y) * (p2.x - p1.x) / (p2.y - p1.y) + p1.x
                    if p1.x == p2.x or point.x <= xinters:
                        inside = not inside
        
        p1 = p2
    
    return inside


def convex_hull(points):
    """
    Calcule l'enveloppe convexe (Graham scan).
    
    Args:
        points: Liste de Points
        
    Returns:
        Liste de Points formant l'enveloppe convexe
        
    Example:
        >>> pts = [Point(0,0), Point(1,1), Point(0,1), Point(1,0), Point(0.5,0.5)]
        >>> hull = convex_hull(pts)
        >>> len(hull)
        4
    """
    if len(points) < 3:
        return points
    
    # Trouver le point le plus bas (puis le plus à gauche)
    start = min(points, key=lambda p: (p.y, p.x))
    
    # Trier par angle polaire
    def polar_angle(p):
        dx = p.x - start.x
        dy = p.y - start.y
        return math.atan2(dy, dx)
    
    sorted_points = sorted([p for p in points if p != start], key=polar_angle)
    
    hull = [start, sorted_points[0]]
    
    for p in sorted_points[1:]:
        while len(hull) > 1 and ccw(hull[-2], hull[-1], p) <= 0:
            hull.pop()
        hull.append(p)
    
    return hull


def closest_pair_bruteforce(points):
    """
    Paire de points la plus proche (bruteforce O(n²)).
    
    Args:
        points: Liste de Points
        
    Returns:
        Tuple (p1, p2, distance)
    """
    n = len(points)
    if n < 2:
        return None
    
    min_dist = float('inf')
    closest = None
    
    for i in range(n):
        for j in range(i + 1, n):
            dist = points[i].distance_to(points[j])
            if dist < min_dist:
                min_dist = dist
                closest = (points[i], points[j], dist)
    
    return closest


def test():
    """Tests unitaires complets"""
    
    # Test Point
    p1 = Point(0, 0)
    p2 = Point(3, 4)
    
    assert p1.distance_to(p2) == 5.0
    assert p1.distance_squared(p2) == 25
    
    # Test opérations vectorielles
    p3 = p1 + p2
    assert p3.x == 3 and p3.y == 4
    
    p4 = p2 - p1
    assert p4.x == 3 and p4.y == 4
    
    # Test produit scalaire
    v1 = Point(1, 0)
    v2 = Point(0, 1)
    assert v1.dot(v2) == 0
    
    # Test produit vectoriel
    assert v1.cross(v2) == 1
    
    # Test CCW
    a = Point(0, 0)
    b = Point(1, 0)
    c = Point(1, 1)
    assert ccw(a, b, c) > 0  # Sens trigo
    
    # Test intersection de segments
    s1_p1 = Point(0, 0)
    s1_p2 = Point(2, 2)
    s2_p1 = Point(0, 2)
    s2_p2 = Point(2, 0)
    assert segments_intersect(s1_p1, s1_p2, s2_p1, s2_p2)
    
    # Test distance point-droite
    point = Point(1, 1)
    line_p1 = Point(0, 0)
    line_p2 = Point(2, 0)
    dist = point_to_line_distance(point, line_p1, line_p2)
    assert abs(dist - 1.0) < 1e-9
    
    # Test aire polygone
    square = [Point(0, 0), Point(1, 0), Point(1, 1), Point(0, 1)]
    area = polygon_area(square)
    assert abs(area - 1.0) < 1e-9
    
    # Test point in polygon
    inside_point = Point(0.5, 0.5)
    outside_point = Point(2, 2)
    assert point_in_polygon(inside_point, square)
    assert not point_in_polygon(outside_point, square)
    
    # Test convex hull
    pts = [Point(0, 0), Point(1, 1), Point(0, 1), Point(1, 0), Point(0.5, 0.5)]
    hull = convex_hull(pts)
    assert len(hull) == 4
    
    # Test closest pair
    pts2 = [Point(0, 0), Point(1, 1), Point(2, 2), Point(10, 10)]
    closest = closest_pair_bruteforce(pts2)
    assert closest[2] < 2.0
    
    print("Tous les tests passes")


if __name__ == "__main__":
    test()

