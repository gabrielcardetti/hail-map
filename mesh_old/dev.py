from collections import defaultdict
from typing import List, Dict
from utils import (
    parse_hail_data, 
    process_polygons, 
    ProcessedPolygonData, 
    HailJSONEncoder,
    LatLng
)
import json
import numpy as np
from shapely.geometry import Polygon
from typing import List, Dict
from utils import LatLng, ProcessedPolygonData

from shapely.geometry import Polygon
from typing import List, Dict
from utils import LatLng, ProcessedPolygonData

def mark_peaks(polygons: List[ProcessedPolygonData]) -> None:
    """
    Mark each polygon with an '_is_peak' attribute.
    A polygon is a peak if it does not contain any polygon with a larger hail size.
    Peaks remain untouched during adjustment.
    """
    for poly in polygons:
        poly['_is_peak'] = True
    for poly in polygons:
        poly_shape = Polygon([(p.lng, p.lat) for p in poly['positions']])
        for other in polygons:
            if other['size'] > poly['size']:
                other_shape = Polygon([(p.lng, p.lat) for p in other['positions']])
                if poly_shape.contains(other_shape):
                    poly['_is_peak'] = False
                    break

def iterative_adjustment(polygons: List[ProcessedPolygonData], min_distance: float = 0.004) -> List[ProcessedPolygonData]:
    """
    Iteratively adjust nested polygons from the innermost (peak) to the outermost region.

    For each adjacent size level (starting from the highest hail size, which are peaks):
      1. For each inner (peak) polygon, copy it and expand it (using a buffer of min_distance).
      2. For every outer polygon in the next size level that overlaps this inner polygon,
         union its geometry with the buffered inner polygon.
      3. The updated outer polygon is then used when processing the next level outward.

    Peaks remain untouched.
    """
    # Step 0: Mark peaks so that peaks remain untouched.
    mark_peaks(polygons)
    
    # Group polygons by hail size.
    polygons_by_size: Dict[int, List[ProcessedPolygonData]] = {}
    for poly in polygons:
        polygons_by_size.setdefault(poly['size'], []).append(poly)
    
    # Sort sizes in descending order (highest hail size = inner/peaks first).
    sizes = sorted(polygons_by_size.keys(), reverse=True)
    
    # Process each adjacent pair: inner (peaks) and outer (to be adjusted).
    for i in range(len(sizes) - 1):
        inner_size = sizes[i]
        outer_size = sizes[i+1]
        inner_group = polygons_by_size[inner_size]
        outer_group = polygons_by_size[outer_size]
        
        # For each inner polygon in the current (peak) level:
        for inner_poly in inner_group:
            inner_coords = [(p.lng, p.lat) for p in inner_poly['positions']]
            try:
                inner_shape = Polygon(inner_coords)
                # Ensure inner shape is valid
                if not inner_shape.is_valid:
                    from shapely.validation import make_valid
                    inner_shape = make_valid(inner_shape)
                
                # For each outer polygon in the next level:
                for outer_poly in outer_group:
                    # Do not adjust if the outer polygon is itself a peak.
                    if outer_poly.get('_is_peak', False):
                        continue
                    
                    outer_coords = [(p.lng, p.lat) for p in outer_poly['positions']]
                    try:
                        outer_shape = Polygon(outer_coords)
                        # Ensure outer shape is valid
                        if not outer_shape.is_valid:
                            from shapely.validation import make_valid
                            outer_shape = make_valid(outer_shape)
                        
                        # If the inner polygon touches or overlaps the outer polygon:
                        if outer_shape.intersects(inner_shape):
                            # Expand the inner polygon with a buffer
                            # Use join_style=1 (mitre) for more precise corners
                            buffered_inner = inner_shape.buffer(min_distance, join_style=1)
                            
                            # Ensure buffered shape is valid
                            if not buffered_inner.is_valid:
                                from shapely.validation import make_valid
                                buffered_inner = make_valid(buffered_inner)
                            
                            # Union the outer polygon with the buffered inner polygon
                            try:
                                new_outer = outer_shape.union(buffered_inner)
                                
                                # If the union returns a MultiPolygon, select the component that contains the buffered inner
                                if new_outer.geom_type == 'MultiPolygon':
                                    candidates = [geom for geom in new_outer.geoms if geom.contains(buffered_inner)]
                                    if candidates:
                                        new_outer = max(candidates, key=lambda g: g.area)
                                    else:
                                        new_outer = max(new_outer.geoms, key=lambda g: g.area)
                                
                                # Update the outer polygon with the new shape
                                outer_poly['positions'] = [LatLng(lat=pt[1], lng=pt[0]) for pt in new_outer.exterior.coords]
                            except Exception as e:
                                print(f"Union operation failed: {e}")
                                # Continue with the original shape
                                continue
                    except Exception as e:
                        print(f"Error processing outer polygon: {e}")
                        continue
            except Exception as e:
                print(f"Error processing inner polygon: {e}")
                continue
    
    return polygons


def smooth_polygon(points: List[LatLng], iterations: int = 20) -> List[LatLng]:
    """
    Smooth a polygon using Chaikin's algorithm.
    
    Args:
        points: List of polygon vertices as LatLng objects
        iterations: Number of smoothing iterations
        
    Returns:
        List of LatLng points forming the smoothed polygon
    """
    if len(points) < 3:
        return points
    
    # Make sure the polygon is closed
    closed = points[0] == points[-1]
    if closed:
        points = points[:-1]  # Remove the closing point temporarily
        
    for _ in range(iterations):
        new_points = []
        n = len(points)
        
        for i in range(n):
            p0 = points[i]
            p1 = points[(i + 1) % n]
            
            # Generate two points per edge
            q0 = LatLng(
                lat=0.75 * p0.lat + 0.25 * p1.lat,
                lng=0.75 * p0.lng + 0.25 * p1.lng
            )
            q1 = LatLng(
                lat=0.25 * p0.lat + 0.75 * p1.lat,
                lng=0.25 * p0.lng + 0.75 * p1.lng
            )
            
            new_points.append(q0)
            new_points.append(q1)
            
        points = new_points
    
    # Re-close the polygon if it was closed
    if closed:
        points.append(points[0])
        
    return points


def smooth_polygons(polygons: List[ProcessedPolygonData], iterations: int = 20) -> List[ProcessedPolygonData]:
    """
    Apply smoothing to all polygons.
    
    Args:
        polygons: List of processed polygon data
        iterations: Number of smoothing iterations
        
    Returns:
        List of smoothed polygon data
    """
    smoothed_polygons = []
    
    for polygon in polygons:
        # Create a new polygon with smoothed positions
        smoothed_polygon = polygon.copy()
        
        if 'positions' in polygon:
            smoothed_polygon['positions'] = smooth_polygon(polygon['positions'], iterations)
        
        smoothed_polygons.append(smoothed_polygon)
    
    return smoothed_polygons


def summarize_hail_data(polygons: List[ProcessedPolygonData]) -> Dict:
    """
    Create a summary of the hail data.

    Args:
        polygons: List of PolygonData objects

    Returns:
        Dictionary containing summary statistics
    """
    summary = {
        'total_regions': len(polygons),
        'size_distribution': defaultdict(int),
        'total_points': 0,
        'size_range': {
            'min': float('inf'),
            'max': float('-inf')
        }
    }

    for polygon in polygons:
        # Count regions by size
        summary['size_distribution'][polygon.size] += 1

        # Count total points
        summary['total_points'] += len(polygon.coordinates)

        # Update size range
        summary['size_range']['min'] = min(
            summary['size_range']['min'], polygon.size)
        summary['size_range']['max'] = max(
            summary['size_range']['max'], polygon.size)

    # Convert defaultdict to regular dict
    summary['size_distribution'] = dict(summary['size_distribution'])

    return summary


def print_summary(summary: Dict) -> None:
    """
    Print a formatted summary of the hail data.

    Args:
        summary: Dictionary containing summary statistics
    """
    print("\nHail Contour Data Summary")
    print("=" * 30)
    print(f"\nTotal Regions: {summary['total_regions']}")
    print(f"Total Points: {summary['total_points']}")
    print(f"\nSize Range:")
    print(
        f"  Minimum: {summary['size_range']['min']}mm ({summary['size_range']['min']/25.4:.2f} inches)")
    print(
        f"  Maximum: {summary['size_range']['max']}mm ({summary['size_range']['max']/25.4:.2f} inches)")

    print("\nSize Distribution:")
    for size, count in sorted(summary['size_distribution'].items()):
        print(f"  {size}mm ({size/25.4:.2f} inches): {count} regions")


def main(file_path: str, smooth: bool = False, smooth_iterations: int = 20, 
         separate_polygons: bool = False, min_distance: float = 1.0) -> None:
    """
    Main function to process hail contour data file.

    Args:
        file_path: Path to the hail contour data file
        smooth: Whether to apply polygon smoothing
        smooth_iterations: Number of smoothing iterations
        separate_polygons: Whether to separate touching polygons
        min_distance: Minimum distance to maintain between polygons
    """
    try:
        # Read the file
        with open(file_path, 'r') as f:
            data = f.read()

        # Parse the data
        polygons = parse_hail_data(data)

        print(f'Polygons: {polygons}')

        # Process the polygons
        processed_polygons = process_polygons(polygons)
        print(f"\nProcessed {len(processed_polygons)} polygons")


        processed_polygons = iterative_adjustment(processed_polygons)

        # Apply smoothing if requested
        processed_polygons = smooth_polygons(processed_polygons, smooth_iterations)
        print(f"Applied {smooth_iterations} smoothing iterations to polygons")
        # save processed_polygons as a .json file
        output_filename = 'processed_polygons_smooth.json' if smooth else 'processed_polygons.json'
        with open(output_filename, 'w') as f:
            json.dump(processed_polygons, f, cls=HailJSONEncoder, indent=2)
            print(f"Saved processed polygons to {output_filename}")

        # Generate and print summary
        summary = summarize_hail_data(polygons)
        print_summary(summary)

    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found")
    except Exception as e:
        print(f"Error processing file: {str(e)}")


if __name__ == "__main__":
    # Example usage
    file_path = "hail_contours.txt"
    main(file_path, smooth=True, smooth_iterations=2)