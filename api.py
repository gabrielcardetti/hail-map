from fastapi import FastAPI, Query
from typing import List, TypedDict
import json
from utils import LatLng, StyleDict
from pipeline import get_connection
import pandas as pd


app = FastAPI()


class ProcessedPolygonData(TypedDict):
    positions: List[LatLng]
    style: StyleDict
    size: int
    inchSize: float
    threshold: int


@app.get("/hail", response_model=dict[str, List[ProcessedPolygonData]])
def read_hail(year: int = Query(2024, description="Year to filter hail data"),
              day: str = Query("0506", description="Day to filter hail data (format: MMDD)")):
    connection = get_connection()
    cursor = connection.cursor(dictionary=True)

    # Query to get hail regions for the specified date
    cursor.execute("""
        SELECT 
            hr.*,
            JSON_ARRAYAGG(
                JSON_OBJECT(
                    'lat', hp.lat,
                    'lng', hp.lng
                )
            ) as positions
        FROM hail_region hr
        LEFT JOIN hail_point hp ON hr.id = hp.hail_region_id
        WHERE hr.date = %s
        GROUP BY hr.id
    """, (f"{year}{day}",))

    regions = cursor.fetchall()

    # Format the data according to ProcessedPolygonData structure
    processed_regions = []
    for region in regions:
        # Create style dictionary
        style: StyleDict = {
            'fillColor': region['fill_color'],
            'fillOpacity': region['fill_opacity'],
            'strokeColor': region['stroke_color'],
            'strokeOpacity': region['stroke_opacity'],
            'strokeWeight': region['stroke_weight']
        }

        # Parse JSON string and convert to LatLng objects
        positions = [LatLng(
            **pos) for pos in json.loads(region['positions'])] if region['positions'] else []

        # Create ProcessedPolygonData
        processed_region = {
            'positions': positions,
            'style': style,
            'size': region['size'],
            'inchSize': region['inch_size'],
            'threshold': region['threshold']
        }
        processed_regions.append(processed_region)

    return {'data': processed_regions}


@app.get("/hail/search")
def search_hail_regions(
    lat: float = Query(32.78689956665039,
                       description="Latitude of the point to search"),
    lng: float = Query(-82.07479858398438,
                       description="Longitude of the point to search")
):
    connection = get_connection()
    cursor = connection.cursor(dictionary=True)

    # Query to find regions that contain the point using the region POLYGON
    cursor.execute("""
        SELECT DISTINCT
            hr.id as regionId,
            hr.date,
            hr.inch_size as inchSize,
            hr.threshold,
            hr.fill_color as fillColor,
            hr.fill_opacity as fillOpacity,
            hr.stroke_color as strokeColor,
            hr.stroke_opacity as strokeOpacity,
            hr.stroke_weight as strokeWeight
        FROM hail_region hr
        WHERE ST_Contains(
            hr.region,
            ST_PointFromText(%s, 4326)
        )
        ORDER BY hr.date DESC
    """, (f'POINT({lng} {lat})',))

    regions = cursor.fetchall()

    # Group by date and keep only the largest inch_size
    date_grouped = {}
    for region in regions:
        date = region['date']
        if date not in date_grouped or region['inchSize'] > date_grouped[date]['inchSize']:
            date_grouped[date] = region

    # Convert decimal values to float for JSON serialization
    regions = []
    for region in date_grouped.values():
        region['fillOpacity'] = float(region['fillOpacity'])
        region['strokeOpacity'] = float(region['strokeOpacity'])
        region['inchSize'] = float(region['inchSize'])
        region['date'] = pd.to_datetime(region['date']).strftime('%Y-%m-%d')
        regions.append(region)

    return {'data': regions}
