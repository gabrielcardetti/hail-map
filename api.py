from fastapi import FastAPI, Query
from typing import List
import json
from utils import ProcessedPolygonData, LatLng, StyleDict
from pipeline import get_connection

app = FastAPI()


@app.get("/hail", response_model=dict[str, List[ProcessedPolygonData]])
def read_hail(year: int = Query(2024, description="Year to filter hail data"),
              day: str = Query("0509", description="Day to filter hail data (format: MMDD)")):
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
        positions = [LatLng(**pos) for pos in json.loads(region['positions'])] if region['positions'] else []
        
        # Create ProcessedPolygonData
        processed_region = {
            'positions': positions,
            'style': style,
            'size': region['size'],
            'inch_size': region['inch_size'],
            'threshold': region['threshold']
        }
        processed_regions.append(processed_region)
    
    return {'data': processed_regions}
