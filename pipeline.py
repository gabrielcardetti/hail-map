from collections import defaultdict
from typing import List, Dict
from utils import (
    parse_hail_data,
    PolygonData,
    process_polygons,
    ProcessedPolygonData,
    HailJSONEncoder
)
import json

import mysql.connector

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


def main(file_path: str, output_path: str, date: str) -> None:
    """
    Main function to process hail contour data file.

    Args:
        file_path: Path to the hail contour data file
        output_path: Path to the output file
        date: Date of the hail contour data
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
        with open(output_path, 'w') as f:
            json.dump(processed_polygons, f, cls=HailJSONEncoder, indent=2)

        # Generate and print summary
        summary = summarize_hail_data(polygons)
        print_summary(summary)

        # insert the polygons into the database
        insert_polygons(processed_polygons, date)

        print("Polygons inserted successfully for date: ", date)

    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found")
    except Exception as e:
        print(f"Error processing file: {str(e)}")


def insert_polygons(polygons: List[ProcessedPolygonData], date: str) -> None:
    """
    Insert processed hail polygons into the database.
    This function handles both the hail region polygons and their constituent points.
    It includes validation and automatic fixing of invalid polygons.

    Args:
        polygons: List of ProcessedPolygonData objects containing polygon information
                 Each polygon has positions (points), style information, and size data
        date: Date string in YYYYMMDD format representing when the hail data was recorded

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
                print(f"Skipping polygon with insufficient points: {len(polygon['positions']) if polygon['positions'] else 0} points")
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
                    print(f"- Size: {polygon['size']}mm ({polygon['inch_size']} inches)")
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
                    print(f"- Longitude range: {min(lngs):.4f} to {max(lngs):.4f}")
                    print(f"- Latitude range: {min(lats):.4f} to {max(lats):.4f}")
                    
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
                        print(f"- Point creation successful: {bool(point_result)}")
                        
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
                            print(f"- Fixed WKT preview: {fixed_wkt[:100]}...")
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
                print(f"Successfully inserted polygon with {len(points)} points")
                
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
                date VARCHAR(50) NOT NULL,
                fill_color VARCHAR(50) NOT NULL,
                fill_opacity FLOAT NOT NULL,
                stroke_color VARCHAR(50) NOT NULL,
                stroke_opacity FLOAT NOT NULL,
                stroke_weight INT NOT NULL,
                size INT NOT NULL,
                inch_size FLOAT NOT NULL,
                threshold INT NOT NULL,
                region POLYGON NOT NULL SRID 4326,
                SPATIAL INDEX(region)
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
    create_tables()

    # Process all files in the @processedFiles folder
    input_folder = "./processedFiles"

    # Create output folder if it doesn't exist
    os.makedirs("processed_json", exist_ok=True)

    # Process each .txt file in the folder
    for file_name in os.listdir(input_folder):
        if file_name.endswith('.txt'):
            input_path = os.path.join(input_folder, file_name)
            # hail_contours_20240924_0000.txt
            # get date from name
            date = file_name.split("_")[2]

            # Create output path with same name but .json extension
            output_path = os.path.join(
                "processed_json", Path(file_name).stem + ".json")

            print(f"\nProcessing: {file_name}")
            main(input_path, output_path, date)
