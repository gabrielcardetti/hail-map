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
    Insert the processed polygons into the database.
    """
    try:
        connection = get_connection()
        cursor = connection.cursor()

        for polygon in polygons:
            # Insert hail region
            region_query = """
                INSERT INTO hail_region (
                    date, fill_color, fill_opacity, stroke_color, 
                    stroke_opacity, stroke_weight, size, inch_size, threshold
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
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
                polygon['threshold']
            )
            cursor.execute(region_query, region_values)
            region_id = cursor.lastrowid

            # Insert points for this region
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
                threshold INT NOT NULL
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
