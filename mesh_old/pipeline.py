from collections import defaultdict
from typing import List, Dict
from datetime import datetime
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
import mysql.connector
import time

host = "localhost"
user = "root"
password = "123"
database = "hail_data"



def get_connection():
    return mysql.connector.connect(
        host=host,
        user=user,
        password=password,
        database=database
    )

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
         separate_polygons: bool = False, min_distance: float = 1.0, output_path: str = None) -> None:
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

        # print(f'Polygons: {polygons}')

        # Process the polygons
        processed_polygons = process_polygons(polygons)
        # print(f"\nProcessed {len(processed_polygons)} polygons")


        processed_polygons = iterative_adjustment(processed_polygons)

        # Apply smoothing if requested
        processed_polygons = smooth_polygons(processed_polygons, smooth_iterations)
        # print(f"Applied {smooth_iterations} smoothing iterations to polygons")
        # # save processed_polygons as a .json file
        # if output_path:
        #     with open(output_path, 'w') as f:
        #         json.dump(processed_polygons, f, cls=HailJSONEncoder, indent=2)
        #     print(f"Saved processed polygons to {output_path}")

        # Generate and print summary
        # summary = summarize_hail_data(polygons)
        # print_summary(summary)


        insert_polygons(processed_polygons, date)

        # print("Polygons inserted successfully for date: ", date)

    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found")
    except Exception as e:
        print(f"Error processing file: {str(e)}")


def insert_polygons(polygons: List[ProcessedPolygonData], date: datetime) -> None:
    """
    Insert processed hail polygons into the database.
    This function handles both the hail region polygons and their constituent points.
    It includes validation and automatic fixing of invalid polygons.

    Args:
        polygons: List of ProcessedPolygonData objects containing polygon information
                 Each polygon has positions (points), style information, and size data
        date: Date object representing when the hail data was recorded

    The function performs the following steps for each polygon:
    1. Validates the polygon has enough points
    2. Creates a Well-Known Text (WKT) representation -> https://wktmap.com/ good place to check the polygon
    3. Validates and attempts to fix invalid geometries
    4. Inserts the polygon into hail_region table
    5. Inserts all points into hail_point table
    """
    try:
        connection = get_connection()
        cursor = connection.cursor()

        for polygon in polygons:
            # Validation Step 1: Check minimum points requirement
            # A valid polygon needs at least 3 points to form an area
            if not polygon['positions'] or len(polygon['positions']) < 3:
                print(
                    f"Skipping polygon with insufficient points: {len(polygon['positions']) if polygon['positions'] else 0} points")
                continue

            # Step 2: Prepare polygon data
            # Convert LatLng objects to (longitude, latitude) pairs
            # Note: MySQL expects longitude first in spatial functions
            points = [(pos.lng, pos.lat) for pos in polygon['positions']]

            # Ensure the polygon is closed (first point equals last point)
            # This is required for valid polygon geometry in MySQL
            if points[0] != points[-1]:
                points.append(points[0])

            # Create Well-Known Text (WKT) representation of the polygon
            # Format: POLYGON((lng1 lat1, lng2 lat2, ...))
            points_str = ','.join(f'{lng} {lat}' for lng, lat in points)
            polygon_wkt = f'POLYGON(({points_str}))'

            try:
                # Step 3: Validate and fix geometry
                # Check if the polygon is valid according to MySQL spatial rules
                cursor.execute("""
                    SELECT 
                        ST_IsValid(ST_GeomFromText(%s, 4326)) as is_valid,
                        ST_NumPoints(ST_GeomFromText(%s, 4326)) as num_points
                """, (polygon_wkt, polygon_wkt))

                validation_result = cursor.fetchone()
                is_valid, num_points = validation_result

                if not is_valid:
                    print(f"\nInvalid polygon details:")
                    print(f"- Number of points: {len(points)}")
                    print(f"- First few points: {points[:3]}")
                    print(f"- Last few points: {points[-3:]}")
                    print(
                        f"- Size: {polygon['size']}mm ({polygon['inch_size']} inches)")
                    print(f"- Threshold: {polygon['threshold']}")

                    # Check for potential data issues
                    has_duplicate_points = len(set(points)) != len(points)
                    has_invalid_coords = any(not (-180 <= lng <= 180 and -90 <= lat <= 90)
                                             for lng, lat in points)

                    print("\nValidation checks:")
                    print(f"- Has duplicate points: {has_duplicate_points}")
                    print(f"- Has invalid coordinates: {has_invalid_coords}")
                    print(f"- Is closed polygon: {points[0] == points[-1]}")

                    # Print the full WKT for inspection
                    print("\nFull WKT representation:")
                    print(polygon_wkt)

                    # Try to identify any coordinate patterns
                    lngs, lats = zip(*points)
                    print("\nCoordinate ranges:")
                    print(
                        f"- Longitude range: {min(lngs):.4f} to {max(lngs):.4f}")
                    print(
                        f"- Latitude range: {min(lats):.4f} to {max(lats):.4f}")

                    # Try simpler geometry first
                    try:
                        # Try creating a point to verify basic spatial operations
                        cursor.execute("""
                            SELECT ST_AsText(
                                ST_PointFromText(%s, 4326)
                            )
                        """, (f'POINT({points[0][0]} {points[0][1]})',))
                        point_result = cursor.fetchone()[0]
                        print("\nBasic spatial operation test:")
                        print(
                            f"- Point creation successful: {bool(point_result)}")

                        # Try to fix the polygon
                        cursor.execute("""
                            SELECT ST_AsText(
                                ST_ConvexHull(ST_GeomFromText(%s, 4326))
                            )
                        """, (polygon_wkt,))
                        fixed_wkt = cursor.fetchone()[0]

                        if fixed_wkt and fixed_wkt.startswith('POLYGON'):
                            print("\nPolygon fix attempt:")
                            print("- Successfully fixed polygon using convex hull")
                            # print(f"- Fixed WKT preview: {fixed_wkt[:100]}...")
                            polygon_wkt = fixed_wkt
                        else:
                            print("\nPolygon fix attempt:")
                            print("- Could not fix polygon")
                            print("- Skipping this polygon")
                            continue
                    except Exception as e:
                        print("\nError details:")
                        print(f"- Error type: {type(e).__name__}")
                        print(f"- Error message: {str(e)}")
                        print(f"- Skipping this polygon")
                        continue

                # Step 4: Insert the hail region
                # Store both the geometric and non-geometric data
                region_query = """
                    INSERT INTO hail_region (
                        date, fill_color, fill_opacity, stroke_color, 
                        stroke_opacity, stroke_weight, size, inch_size, threshold,
                        region
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, ST_GeomFromText(%s, 4326))
                """
                region_values = (
                    date,
                    polygon['style']['fillColor'],
                    polygon['style']['fillOpacity'],
                    polygon['style']['strokeColor'],
                    polygon['style']['strokeOpacity'],
                    polygon['style']['strokeWeight'],
                    polygon['size'],
                    polygon['inch_size'],
                    polygon['threshold'],
                    polygon_wkt
                )
                cursor.execute(region_query, region_values)
                region_id = cursor.lastrowid

                # Step 5: Insert individual points
                # Store each point with a spatial index for efficient querying
                points_query = """
                    INSERT INTO hail_point (
                        hail_region_id, lat, lng, point
                    ) VALUES (%s, %s, %s, ST_PointFromText(%s, 4326))
                """

                points_values = [
                    (
                        region_id,
                        position.lat,
                        position.lng,
                        f'POINT({position.lng} {position.lat})'
                    )
                    for position in polygon['positions']
                ]

                cursor.executemany(points_query, points_values)
                # print(f"Successfully inserted polygon with {len(points)} points")

            except mysql.connector.Error as error:
                print(f"Failed to insert polygon: {error}")
                print(f"Polygon WKT preview: {polygon_wkt[:100]}...")
                continue

        connection.commit()
        print(f"Successfully inserted {len(polygons)} polygons")

    except mysql.connector.Error as error:
        print(f"Failed to insert polygons: {error}")
        if connection.is_connected():
            connection.rollback()

    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()


def create_tables() -> None:
    """
    Create tables for storing the processed polygons if they don't exist.
    """
    try:
        # Connect to MySQL
        print("Creating tables...")
        connection = get_connection()
        cursor = connection.cursor()

        # Create hail_regions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS hail_region (
                id INT AUTO_INCREMENT PRIMARY KEY,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                date DATE NOT NULL,
                fill_color VARCHAR(50) NOT NULL,
                fill_opacity FLOAT NOT NULL,
                stroke_color VARCHAR(50) NOT NULL,
                stroke_opacity FLOAT NOT NULL,
                stroke_weight INT NOT NULL,
                size INT NOT NULL,
                inch_size FLOAT NOT NULL,
                threshold INT NOT NULL,
                region POLYGON NOT NULL SRID 4326,
                SPATIAL INDEX(region),
                INDEX idx_date (date),
                INDEX idx_date_inch_size (date, inch_size)
            )
        """)

        # Create hail_point table with spatial point
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS hail_point (
                id INT AUTO_INCREMENT PRIMARY KEY,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                hail_region_id INT NOT NULL,
                lat FLOAT NOT NULL,
                lng FLOAT NOT NULL,
                point POINT NOT NULL SRID 4326,
                FOREIGN KEY (hail_region_id) REFERENCES hail_region(id),
                SPATIAL INDEX(point)
            )
        """)

        connection.commit()
        print("Tables created successfully")

    except mysql.connector.Error as error:
        print(f"Failed to create tables: {error}")


def clean_tables() -> None:
    """
    Clean the tables.
    """

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("DELETE FROM hail_point WHERE id > 0")
    cursor.execute("DELETE FROM hail_region WHERE id > 0")

    connection.commit()
    print("Tables cleaned successfully")


if __name__ == "__main__":
    import os
    from pathlib import Path

    # clean_tables()
    # create_tables()

    # # Process all files in the @processedFiles folder
    input_folder = "./contours-kgsp-kcae-2025-until-march-11"

    # Create output folder if it doesn't exist
    os.makedirs("processed_json_march-11-2", exist_ok=True)

    # Process each .txt file in the folder
    for file_name in os.listdir(input_folder):
        if file_name.endswith('.txt'):
            start_time = time.time()
            # input_path = os.path.join(input_folder, file_name)
            input_path = input_folder + '/' +file_name
            # hail_contours_KCAE_20210503_0000.txt
            # get date from name
            date = file_name.split("_")[3]
            date = datetime.strptime(date, "%Y%m%d")

            # Create output path with same name but .json extension
            output_path = os.path.join(
                "processed_json_march-11-2", Path(file_name).stem + ".json")

            print(f"\nProcessing: {file_name}")
            # main(input_path, output_path, date)
            main(input_path, smooth=True, smooth_iterations=2, output_path=output_path)
            print(f"Time taken: {time.time() - start_time:.2f} seconds for {file_name}")

