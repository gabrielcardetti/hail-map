import numpy as np
import math
import json
from scipy.spatial import Delaunay

from typing import TypedDict, List, Dict, Union, Set, Tuple
from dataclasses import dataclass


class HailJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for hail data classes."""

    def default(self, obj):
        if isinstance(obj, LatLng):
            return {'lat': obj.lat, 'lng': obj.lng}
        return super().default(obj)


class StyleDict(TypedDict):
    fillColor: str
    fillOpacity: float
    strokeColor: str
    strokeOpacity: float
    strokeWeight: Union[int, float]


@dataclass
class LatLng:
    lat: float
    lng: float


@dataclass
class PolygonData:
    coordinates: List[LatLng]
    style: StyleDict
    size: int
    inch_size: float
    threshold: int


class ProcessedPolygonData(TypedDict):
    positions: List[LatLng]
    style: StyleDict
    size: int
    inch_size: float
    threshold: int


# Constants
HAIL_STYLE_MAP: Dict[int, StyleDict] = {
    19: {  # 0.75 inch
        'fillColor': '#FFF9D5',
        'fillOpacity': 0.5,
        'strokeColor': '#FFF9D5',
        'strokeOpacity': 1,
        'strokeWeight': 1,
        'inch_size': 0.75
    },
    25: {  # 1.0 inch
        'fillColor': '#E0EE98',
        'fillOpacity': 0.5,
        'strokeColor': '#E0EE98',
        'strokeOpacity': 1,
        'strokeWeight': 1,
        'inch_size': 1.0
    },
    32: {  # 1.25 inch
        'fillColor': '#FFDE27',
        'fillOpacity': 0.5,
        'strokeColor': '#FFDE27',
        'strokeOpacity': 1,
        'strokeWeight': 1,
        'inch_size': 1.25
    },
    38: {  # 1.5 inch
        'fillColor': '#FEAE0E',
        'fillOpacity': 0.5,
        'strokeColor': '#FEAE0E',
        'strokeOpacity': 1,
        'strokeWeight': 1.5,
        'inch_size': 1.5
    },
    44: {  # 1.75 inch
        'fillColor': '#ED6F2D',
        'fillOpacity': 0.5,
        'strokeColor': '#ED6F2D',
        'strokeOpacity': 1,
        'strokeWeight': 1.5,
        'inch_size': 1.75
    },
    51: {  # 2.0 inch
        'fillColor': '#E94025',
        'fillOpacity': 1,
        'strokeColor': '#E94025',
        'strokeOpacity': 1,
        'strokeWeight': 2,
        'inch_size': 2.0
    },
    64: {  # 2.5 inch
        'fillColor': '#9C2740',
        'fillOpacity': 0.45,
        'strokeColor': '#9C2740',
        'strokeOpacity': 0.65,
        'strokeWeight': 2,
        'inch_size': 2.5
    },
    76: {  # 3.0 inch
        'fillColor': '#673A37',
        'fillOpacity': 0.45,
        'strokeColor': '#673A37',
        'strokeOpacity': 0.65,
        'strokeWeight': 2.5,
        'inch_size': 3.0
    },
    102: {  # 5.0+ inch
        'fillColor': '#3F51B5',
        'fillOpacity': 0.5,
        'strokeColor': '#3F51B5',
        'strokeOpacity': 0.7,
        'strokeWeight': 3,
        'inch_size': 5.0
    }
}


def parse_hail_data(data: str) -> List[PolygonData]:
    """
    Parse hail contour data from a string into a list of polygon data.

    Args:
        data: String containing the hail contour data in the specified format

    Returns:
        List of PolygonData objects containing the parsed polygons sorted by size
    """
    polygons: List[PolygonData] = []
    current_threshold: Union[int, None] = None
    current_points: List[LatLng] = []

    lines = data.split('\n')

    for line in lines:
        trimmed_line = line.strip()

        if not trimmed_line:
            if current_points:
                polygons.append(PolygonData(
                    coordinates=current_points.copy(),
                    style=HAIL_STYLE_MAP[current_threshold or 19],
                    size=current_threshold or 19,
                    inch_size=HAIL_STYLE_MAP[current_threshold or 19]['inch_size'],
                    threshold=current_threshold or 19
                ))
                current_points = []
            continue

        # Match threshold lines (e.g. "0.75inch (threshold: 19mm)")
        import re
        threshold_match = re.match(
            r'(\d+(?:\.\d+)?)\s*inch\s*\(threshold:\s*(\d+)mm\)', trimmed_line)
        if threshold_match:
            current_threshold = int(threshold_match.group(2))  # Use mm value
            continue

        # Match coordinate lines (e.g. "34.8048,-82.0713")
        coord_match = re.match(r'^(\d+\.\d+),(-\d+\.\d+)$', trimmed_line)
        if coord_match:
            current_points.append(LatLng(
                lat=float(coord_match.group(1)),
                lng=float(coord_match.group(2))
            ))
            continue

        # Match region headers (e.g. "Region 1")
        if re.match(r'^Region \d+$', trimmed_line):
            if current_points:
                polygons.append(PolygonData(
                    coordinates=current_points.copy(),
                    style=HAIL_STYLE_MAP[current_threshold or 19],
                    size=current_threshold or 19,
                    inch_size=HAIL_STYLE_MAP[current_threshold or 19]['inch_size'],
                    threshold=current_threshold or 19
                ))
                current_points = []
            continue

    # Don't forget to add the last region if exists
    if current_points:
        polygons.append(PolygonData(
            coordinates=current_points.copy(),
            style=HAIL_STYLE_MAP[current_threshold or 19],
            size=current_threshold or 19,
            inch_size=HAIL_STYLE_MAP[current_threshold or 19]['inch_size'],
            threshold=current_threshold or 19
        ))

    # Sort polygons by size (smallest first)
    return sorted(polygons, key=lambda x: x.size)


def compute_convex_hull(points: List[LatLng]) -> List[LatLng]:
    """
    Compute convex hull using Graham scan algorithm.

    Args:
        points: List of LatLng points

    Returns:
        List of LatLng points forming the convex hull
    """
    # Find point with lowest latitude
    start_point = min(points, key=lambda p: (p.lat, p.lng))

    def cross_product(p1: LatLng, p2: LatLng, p3: LatLng) -> float:
        return ((p2.lng - p1.lng) * (p3.lat - p1.lat) -
                (p2.lat - p1.lat) * (p3.lng - p1.lng))

    # Sort points by angle and distance from start_point
    sorted_points = sorted(
        [p for p in points if p != start_point],
        key=lambda p: (
            math.atan2(p.lat - start_point.lat, p.lng - start_point.lng),
            math.hypot(p.lat - start_point.lat, p.lng - start_point.lng)
        )
    )

    hull = [start_point]
    for point in sorted_points:
        while len(hull) >= 2 and cross_product(hull[-2], hull[-1], point) <= 0:
            hull.pop()
        hull.append(point)

    # Close the polygon
    hull.append(start_point)
    return hull


def compute_circumcircle(p1: LatLng, p2: LatLng, p3: LatLng) -> Tuple[LatLng, float]:
    """
    Compute circumcenter and radius of triangle.

    Args:
        p1, p2, p3: Triangle vertices as LatLng

    Returns:
        Tuple of (circumcenter as LatLng, radius)
    """
    x1, y1 = p1.lng, p1.lat
    x2, y2 = p2.lng, p2.lat
    x3, y3 = p3.lng, p3.lat

    d = 2 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    if d == 0:
        return LatLng(lat=float('inf'), lng=float('inf')), float('inf')

    ux = ((x1 * x1 + y1 * y1) * (y2 - y3) +
          (x2 * x2 + y2 * y2) * (y3 - y1) +
          (x3 * x3 + y3 * y3) * (y1 - y2)) / d

    uy = ((x1 * x1 + y1 * y1) * (x3 - x2) +
          (x2 * x2 + y2 * y2) * (x1 - x3) +
          (x3 * x3 + y3 * y3) * (x2 - x1)) / d

    radius = math.sqrt((x1 - ux) ** 2 + (y1 - uy) ** 2)
    return LatLng(lat=uy, lng=ux), radius


def process_polygon_coordinates(points: List[LatLng], alpha: float = 0.025) -> List[LatLng]:
    """
    Process polygon coordinates using alpha shape algorithm.

    Args:
        points: List of LatLng coordinates
        alpha: Maximum allowed circumradius

    Returns:
        List of LatLng points forming the processed polygon
    """
    if len(points) < 9:
        return []

    if len(points) < 3:
        return points

    # # Filter points that are too close to each other
    # filtered_points = []
    # min_distance = 0.005  # Minimum distance between points

    # for point in points:
    #     if not filtered_points or all(
    #         math.hypot(p.lat - point.lat, p.lng - point.lng) >= min_distance 
    #         for p in filtered_points
    #     ):
    #         filtered_points.append(point)

    # points = filtered_points

    # Convert to numpy array for Delaunay triangulation
    coords = np.array([[p.lng, p.lat] for p in points])
    delaunay = None
    triangle_indices = None
    try:
        delaunay = Delaunay(coords)
        triangle_indices = delaunay.simplices
    except Exception:
        return compute_convex_hull(points)

    # Edge counting map
    edge_map: Dict[Tuple[int, int], int] = {}

    def add_edge(i: int, j: int) -> None:
        edge = tuple(sorted([i, j]))
        edge_map[edge] = edge_map.get(edge, 0) + 1

    # Process triangles
    for simplex in triangle_indices:
        i, j, k = simplex
        p1, p2, p3 = [points[idx] for idx in [i, j, k]]
        center, radius = compute_circumcircle(p1, p2, p3)

        if radius <= alpha:
            add_edge(i, j)
            add_edge(j, k)
            add_edge(k, i)

    # Build adjacency map for boundary edges
    adj: Dict[int, Set[int]] = {}
    for (i, j), count in edge_map.items():
        if count == 1:
            if i not in adj:
                adj[i] = set()
            if j not in adj:
                adj[j] = set()
            adj[i].add(j)
            adj[j].add(i)

    if not adj:
        return compute_convex_hull(points)

    # Build boundary polygon
    boundary: List[int] = []
    start = next(iter(adj.keys()))
    current = start
    prev = -1

    for _ in range(len(points)):
        boundary.append(current)
        if current not in adj:
            break

        neighbors = adj[current]
        if not neighbors:
            break

        next_point = next(iter(neighbors))
        if len(neighbors) == 2:
            next_point = next(p for p in neighbors if p != prev)

        if next_point == start:
            break

        prev = current
        current = next_point

    if len(boundary) < 3:
        return compute_convex_hull(points)

    # Convert indices back to points and close the polygon
    result = [points[i] for i in boundary]
    result.append(points[start])
    return result


def process_polygons(polygons: List[PolygonData]) -> List[ProcessedPolygonData]:
    """
    Process all polygons in the data.

    Args:
        polygons: List of PolygonData objects

    Returns:
        List of processed polygon data with positions, style, and size
    """
    return [{
        'positions': process_polygon_coordinates(polygon.coordinates),
        'style': polygon.style,
        'size': polygon.size,
        'inch_size': polygon.inch_size,
        'threshold': polygon.threshold
    } for polygon in polygons]
