import numpy as np
import pandas as pd
import nexradaws
import pyart
from mesh_grid import main

def get_temperature_levels():
    freezing_level = 3200
    neg20_level = 6500
    return [freezing_level, neg20_level]

def process_radar_volume(radar):
    # Create 3D grid from radar sweeps
    grid = pyart.map.grid_from_radars(
        (radar,),
        grid_shape=(20, 500, 500),  # vertical, horizontal points
        grid_limits=((0, 20000), (-150000, 150000), (-150000, 150000)),  # meters
    )
    
    # Get temperature levels
    levels = get_temperature_levels()
    
    # Calculate MESH and related fields
    mesh_data = main(
        grid=grid,
        dbz_fname='reflectivity',
        levels=levels,
        radar_band='S',  # NEXRAD is S-band
        mesh_method='mh2019_75'  # More modern calibration
    )
    
    return mesh_data

def sort_boundary_points(points):
    """Sort boundary points in clockwise order around their centroid."""
    # Calculate centroid
    centroid = points.mean(axis=0)
    
    # Calculate angles from centroid to each point
    angles = np.arctan2(points[:,0] - centroid[0], 
                       points[:,1] - centroid[1])
    
    # Sort points by angle
    sorted_indices = np.argsort(angles)
    return points[sorted_indices]

def get_hail_bands(mesh_data, lat, lon, min_distance_km=2):
    """Calculate hail size bands from MESH data."""
    thresholds = {
        "0.75inch": 19,
        "1inch": 25,
        "1.5inch": 38,
        "2inch": 51,
        "2.5inch": 64,
        "3inch": 76,
        "4inch": 102
    }
    
    # Open output file
    with open('hail_contours.txt', 'w') as f:
        f.write("Hail Contour Data\n")
        f.write("=================\n\n")
        
        bands = {}
        mesh = mesh_data[0]
        grid_res_km = 0.6
        structure_size = max(1, int(min_distance_km / grid_res_km))
        
        for name, threshold in thresholds.items():
            mask = mesh > threshold
            if np.any(mask):
                structure = np.ones((structure_size, structure_size))
                
                from scipy import ndimage
                dilated = ndimage.binary_dilation(mask, structure=structure)
                eroded = ndimage.binary_erosion(dilated, structure=structure)
                labeled_array, num_regions = ndimage.label(eroded, structure=np.ones((3,3)))
                
                boundary_points_list = []
                
                f.write(f"\n{name} (threshold: {threshold}mm)\n")
                f.write("-" * 40 + "\n")
                
                for region_num in range(1, num_regions + 1):
                    region_mask = labeled_array == region_num
                    gradient_y, gradient_x = np.gradient(region_mask.astype(float))
                    boundary_mask = (gradient_x**2 + gradient_y**2) > 0
                    
                    if np.any(boundary_mask):
                        y_idx, x_idx = np.where(boundary_mask)
                        boundary_points = np.column_stack((
                            lat[y_idx, x_idx],
                            lon[y_idx, x_idx]
                        ))
                        
                        boundary_points = sort_boundary_points(boundary_points)
                        boundary_points_list.append(boundary_points)
                        
                        f.write(f"\nRegion {region_num}\n")
                        f.write(f"Points: {len(boundary_points)}\n")
                        f.write("lat,lon\n")
                        for point in boundary_points:
                            f.write(f"{point[0]:.4f},{point[1]:.4f}\n")
                
                if boundary_points_list:
                    bands[name] = {
                        "threshold_mm": threshold,
                        "boundary_points": boundary_points_list
                    }
    
    return bands

def main_loop():
    templocation = "./files"
    radar_id = 'KGSP'
    start = pd.Timestamp(2023, 5, 9, 18).tz_localize('UTC')
    end = pd.Timestamp(2023, 5, 9, 22).tz_localize('UTC')
    
    conn = nexradaws.NexradAwsInterface()
    scans = conn.get_avail_scans_in_range(start, end, radar_id)
    print(f"Found {len(scans)} scans")
    
    results = conn.download(scans, templocation)
    
    # Initialize accumulation grid
    accumulated_mesh = None
    grid_lat = None
    grid_lon = None
    
    # First pass: accumulate maximum MESH values
    for scan in results.iter_success():
        if scan.filename[-3:] != "MDM":
            print(f"\nProcessing: {scan.filename}")
            radar = scan.open_pyart()
            
            mesh_output = process_radar_volume(radar)
            mesh = mesh_output['mesh_mh2019_75']['data']
            
            if accumulated_mesh is None:
                accumulated_mesh = mesh[0]  # Initialize with first scan
                
                # Create grid coordinates (only need to do this once)
                x = np.linspace(-150000, 150000, 500)
                y = np.linspace(-150000, 150000, 500)
                X, Y = np.meshgrid(x, y)
                
                radar_lat = radar.latitude['data'][0]
                radar_lon = radar.longitude['data'][0]
                
                grid_lat = radar_lat + (Y/111000)
                grid_lon = radar_lon + (X/(111000*np.cos(np.radians(radar_lat))))
            else:
                # Update with maximum values
                accumulated_mesh = np.maximum(accumulated_mesh, mesh[0])
    
    # Now create bands from accumulated data
    print("\nGenerating final hail swath bands...")
    bands = get_hail_bands([accumulated_mesh], grid_lat, grid_lon)
    
    return bands

if __name__ == "__main__":
    hail_bands = main_loop()


