from collections import defaultdict
from typing import List, Dict
from utils import (
    parse_hail_data, 
    process_polygons, 
    ProcessedPolygonData, 
    HailJSONEncoder
)
import json


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


def main(file_path: str) -> None:
    """
    Main function to process hail contour data file.

    Args:
        file_path: Path to the hail contour data file
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

        # save processed_polygons as a .json file
        with open('processed_polygons.json', 'w') as f:
            json.dump(processed_polygons, f, cls=HailJSONEncoder, indent=2)

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
    main(file_path)