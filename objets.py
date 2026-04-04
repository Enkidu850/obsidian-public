class Point:
    def __init__(self, x, y, epsg=4326):
        self.x = x
        self.y = y
        self.epsg = epsg

    def __repr__(self):
        return f"({self.x}, {self.y}) [EPSG:{self.epsg}]"

    def distance_to(self, other):
        if self.epsg != other.epsg:
            raise ValueError("Les points doivent être dans le même système de coordonnées (EPSG).")
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5

class Line:
    def __init__(self, points, epsg=4326):
        if len(points) < 2:
            raise ValueError("Une LINE doit contenir au moins deux points.")
        self.points = points
        self.epsg = epsg

    def __repr__(self):
        return f"({', '.join(str(p) for p in self.points)})"

    def length(self):
        from pyproj import Transformer
        transformer = Transformer.from_crs("EPSG:"+str(self.epsg), "EPSG:4326", always_xy=True)
        return sum(self.points[i].distance_to(self.points[i+1]) for i in range(len(self.points) - 1))

class Polygon:
    def __init__(self, points, epsg=4326):
        if len(points) < 3:
            raise ValueError("Un POLYGON doit contenir au moins trois sommets.")
        self.points = points
        if points[0] != points[-1]:
            self.points.append(points[0])  # fermeture automatique
        self.epsg = epsg

    def __repr__(self):
        return f"({', '.join(str(p) for p in self.points)})"

    def perimeter(self):
        return sum(self.points[i].distance_to(self.points[i+1]) for i in range(len(self.points) - 1))
    
    def area(self):
        # Algorithme de l'aire de Shoelace
        n = len(self.points)
        area = 0.0
        for i in range(n - 1):
            x1, y1 = self.points[i].x, self.points[i].y
            x2, y2 = self.points[i + 1].x, self.points[i + 1].y
            area += (x1 * y2) - (x2 * y1)
        return abs(area) / 2.0