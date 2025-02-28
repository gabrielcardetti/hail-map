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


def smooth_polygon(points: List[LatLng], iterations: int = 2) -> List[LatLng]:
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


def smooth_polygons(polygons: List[ProcessedPolygonData], iterations: int = 2) -> List[ProcessedPolygonData]:
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


def main(file_path: str, smooth: bool = False, smooth_iterations: int = 2) -> None:
    """
    Main function to process hail contour data file.

    Args:
        file_path: Path to the hail contour data file
        smooth: Whether to apply polygon smoothing
        smooth_iterations: Number of smoothing iterations
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