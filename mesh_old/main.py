# main.py (modified)
import numpy as np
import pandas as pd
import pyart
import nexradaws
from scipy import ndimage
from mesh_grid import main as hail_mesh  # Renamed import for clarity
import os
import time
from typing import List
from scipy.ndimage import binary_erosion

# Hail size thresholds (in mm) corresponding to various size bands
thresholds = {
    "0.75inch": 19,
    "1inch": 25,
    "1.2inch": 32,
    "1.5inch": 38,
    "1.75inch": 44,
    "2inch": 51,
    "2.25inch": 57,
    "2.5inch": 64,
    "3inch":   76,
    "4inch":   102
}


def get_temperature_levels():
    """
    Provide environmental temperature levels for hail algorithm.
    Returns [freezing_level, negative20C_level] in meters.
    """
    # In a real-world scenario, these could be fetched from a sounding or model data.
    freezing_level = 3200   # 0°C level (default 3.2 km, can adjust per environment)
    neg20_level    = 6500   # -20°C level (default 6.5 km)
    return [freezing_level, neg20_level]

def process_radar_volume(radar):
    """
    Grid a single radar volume to Cartesian coordinates and compute hail metrics.
    Returns a dictionary with MESH and related fields.
    """
    
    # Define grid size and extent
    grid_shape = (20, 700, 700)  # (vertical levels, y, x)
    grid_limits = ((0, 20000),    # 0 to 20 km altitude
                   (-200000, 200000),  # y: -150 to +150 km
                   (-200000, 200000))  # x: -150 to +150 km

    # Use a tighter radius and Cressman weighting to avoid oversmoothing hail cores
    radar_fields = ['reflectivity']
    
    grid = pyart.map.grid_from_radars(
        [radar],
        grid_shape=grid_shape,
        grid_limits=grid_limits,
        fields=radar_fields,
        weighting_function="Cressman",
        constant_roi=1000.0,         # 1 km radius of influence for interpolation
        roi_func='constant'          # use constant radius (override default dist_beam)
    )
    
    levels = get_temperature_levels()
    print(f"\nUsing temperature levels: {levels}")
    
    mesh_data = hail_mesh(
        grid=grid,
        dbz_fname='reflectivity',
        levels=levels,
        radar_band='S',         # NEXRAD is S-band radar
        mesh_method='mh2019_75' # use modern calibration for MESH (75th percentile)
    )
    
    return mesh_data

def sort_boundary_points(points):
    """Sort boundary points in clockwise order around their centroid."""
    if points.size == 0:
        return points
    centroid = points.mean(axis=0)
    angles = np.arctan2(points[:, 0] - centroid[0],
                        points[:, 1] - centroid[1])
    sorted_indices = np.argsort(angles)
    return points[sorted_indices]

def polynomical_correction(n):
    """
    Applies a thresholded correction to hail size estimations
    (supports scalars and NumPy arrays).
    
    Parameters
    ----------
    n : array_like
        Input array of values to correct
    radar_id : str, optional
    """
    n = np.asarray(n)
    constant = 24
    multiplier = -0.22

    exponent = 0
    max_correction = 30

    
    correction = np.minimum(constant + (multiplier * n) + np.power(n, exponent), max_correction)
    corrected = np.maximum(n - correction, 0)
    return corrected

def blind_cone_correction(mesh, lat_grid, lon_grid, radar_lat=34.883306, radar_lon=-82.219833):
    """
    Applies a correction to account for radar blind cone effects.
    Increases MESH values by a multiplier that decreases linearly within 0.2 degrees of the radar location.
    
    Parameters
    ----------
    mesh : np.ndarray
        Input MESH grid to correct
    lat_grid : np.ndarray
        Grid of latitude values
    lon_grid : np.ndarray
        Grid of longitude values
    radar_lat : float
        Radar latitude in degrees
    radar_lon : float
        Radar longitude in degrees
    
    Returns
    -------
    np.ndarray
        Corrected MESH grid
    """
    # Calculate distance from each point to radar (in degrees)
    dist_lat = lat_grid - radar_lat
    dist_lon = lon_grid - radar_lon
    distance = np.sqrt(dist_lat**2 + dist_lon**2)
    
    # Create correction mask (0.2 degrees radius)
    max_distance = 0.16
    correction_mask = distance <= max_distance
    
    # Calculate linear multiplier (2.0 at radar location, 1.0 at 0.2 degrees)
    multiplier = np.ones_like(mesh)
    multiplier[correction_mask] = 2.2 - (distance[correction_mask] / max_distance)
    
    # Apply correction
    corrected_mesh = mesh * multiplier
    
    return corrected_mesh


def get_hail_bands(mesh_data, lat, lon, min_distance_km=3.5, output_file=None):
    """
    Calculate hail size bands from MESH data.
    Returns a dict of hail bands with their polygon boundary points.
    
    Args:
        mesh_data: MESH grid data
        lat: latitude grid
        lon: longitude grid
        min_distance_km: controls the gap closure for polygon merging
        output_file: path to output file (optional)
    """

    # Analyze mesh data statistics
    mesh = mesh_data[0]  # 2D MESH field
    print(f"[DEBUG] Original MESH - Min: {np.nanmin(mesh):.1f}, Max: {np.nanmax(mesh):.1f}")
    
    # Apply smoothing before corrections
    print(f"[DEBUG] Smoothed MESH - Min: {np.nanmin(mesh):.1f}, Max: {np.nanmax(mesh):.1f}")
    
    # Apply both corrections
    corrected_mesh = polynomical_correction(mesh)
    corrected_mesh = blind_cone_correction(corrected_mesh, lat, lon)
    print(f"[DEBUG] Corrected MESH - Min: {np.nanmin(corrected_mesh):.1f}, Max: {np.nanmax(corrected_mesh):.1f}")

    # Modify file opening to use the output_file parameter
    if output_file:
        f = open(output_file, 'w')
        print(f"[DEBUG] Writing output to {output_file}")
    else:
        f = open('hail_contours.txt', 'w')
        print(f"[DEBUG] Writing output to hail_contours.txt")

    try:
        f.write("Hail Contour Data\n")
        f.write("=================\n\n")

        bands = {}
        grid_res_km = 0.6    # horizontal grid resolution in km (600 m)
        # Compute structuring element size (in pixels) for morphological operations
        structure_size = max(1, int(min_distance_km / grid_res_km))
        print(f"[DEBUG] Structure element size for morphology: {structure_size}x{structure_size} pixels")
        structure = np.ones((structure_size, structure_size), dtype=bool)

        # Identify hail regions for each threshold
        for name, threshold in thresholds.items():
            # Apply correction to mesh values
            
            mask = corrected_mesh > threshold
            
            if not np.any(mask):
                print(f"[DEBUG] No hail of size {name} (threshold {threshold}mm) found")
                continue  # no hail of this size
                
            # Morphological filtering: close small gaps within `min_distance_km`
            print(f"[DEBUG] Applying morphological operations")
            dilated = ndimage.binary_dilation(mask, structure=structure)
            filled = ndimage.binary_erosion(dilated, structure=structure)

            # Now label connected hail areas using the eroded mask
            labeled_array, num_regions = ndimage.label(filled, structure=np.ones((3,3), dtype=int))
            boundary_points_list = []

            f.write(f"\n{name} (threshold: {threshold}mm)\n")
            f.write("-" * 40 + "\n")
            
            for region_idx in range(1, num_regions + 1):
                region_mask = (labeled_array == region_idx)
                region_size = np.sum(region_mask)
                
                # Skip tiny regions (noise) by area – require at least ~4 pixels (~0.86 km^2)
                if region_size < 4:
                    continue
                    
                # Find boundary by looking at the gradient (edge) of the region mask
                grad_y, grad_x = np.gradient(region_mask.astype(float))
                boundary_mask = (grad_x**2 + grad_y**2) > 0
                
                if not np.any(boundary_mask):
                    continue
                    
                # Get coordinates of boundary points
                y_idx, x_idx = np.where(boundary_mask)
                
                # Check if indices are within bounds
                if np.any(y_idx >= lat.shape[0]) or np.any(x_idx >= lat.shape[1]):
                    # Filter out-of-bounds indices
                    valid_idx = (y_idx < lat.shape[0]) & (x_idx < lat.shape[1])
                    y_idx = y_idx[valid_idx]
                    x_idx = x_idx[valid_idx]
                    if len(y_idx) == 0:
                        continue
                
                # Collect lat-lon coordinates of the boundary
                boundary_points = np.column_stack((lat[y_idx, x_idx], lon[y_idx, x_idx]))
                
                # Sort boundary points
                boundary_points = sort_boundary_points(boundary_points)
                
                boundary_points_list.append(boundary_points)
                f.write(f"\nRegion {region_idx}\n")
                f.write(f"Points: {len(boundary_points)}\n")
                f.write("lat,lon\n")
                for point in boundary_points:
                    f.write(f"{point[0]:.4f},{point[1]:.4f}\n")

            if boundary_points_list:
                bands[name] = {
                    "threshold_mm": threshold,
                    "boundary_points": boundary_points_list
                }
            else:
                print(f"[DEBUG] No valid regions found for band '{name}'")
                
        # Log final results
        print(f"\n[DEBUG] Final bands dictionary has {len(bands)} entries")
            
        return bands

    except Exception as e:
        import traceback
        print(f"[ERROR] Exception in get_hail_bands: {type(e).__name__}: {str(e)}")
        traceback.print_exc()
        return {}
        
    finally:
        f.close()
  

def get_grid_filename(start, end, radar_ids):
    """
    Create a standardized filename for grid data.
    
    Args:
        start: Start time for processing
        end: End time for processing
        radar_ids: List of radar station identifiers
    
    Returns:
        String with the grid filename
    """
    radar_str = '_'.join(radar_ids) if isinstance(radar_ids, list) else radar_ids
    time_str = f"{start.strftime('%Y%m%d_%H%M')}_{end.strftime('%Y%m%d_%H%M')}"
    return f"grid_mesh_{radar_str}_{time_str}.npz"

def main_loop(
    start: pd.Timestamp = pd.Timestamp(2023, 5, 9, 10, tz='UTC'),
    end: pd.Timestamp = pd.Timestamp(2023, 5, 10, 2, tz='UTC'),
    radar_ids: List[str] = ['KCAE', 'KGSP'],
    temp_dir: str = "./files",
    output_file: str = None,
    grid_dir: str = "./grids",
    min_distance_km: float = 4,
    grid_center_lat: float = 30.26,
    grid_center_lon: float = -97.70
) -> dict:
    """
    Process radar data for a given time range and radar(s).
    
    Args:
        start: Start time for processing
        end: End time for processing
        radar_ids: List of radar station identifiers
        temp_dir: Directory for temporary files
        output_file: Path for output file
        grid_dir: Directory for saved grid files
        min_distance_km: Distance parameter for hail band calculation
        grid_center_lat: Latitude of the grid center
        grid_center_lon: Longitude of the grid center
    
    Returns:
        Dictionary with hail band information
    """
    # Create directories if they don't exist
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(grid_dir, exist_ok=True)
    
    # Create grid filename
    grid_filename = get_grid_filename(start, end, radar_ids)
    grid_path = os.path.join(grid_dir, grid_filename)
    
    try:
        # Check if grid data exists
        if os.path.exists(grid_path):
            print(f"Loading existing grid data from {grid_path}")
            grid_data = np.load(grid_path)
            accumulated_mesh = grid_data['grid_mesh']
            grid_lat = grid_data['grid_lat']
            grid_lon = grid_data['grid_lon']
        else:
            # Process radar volumes
            print(f"Processing radar data for {', '.join(radar_ids)} from {start} to {end}")
            
            # Hardcoded grid parameters - centered approximately between KGSP and KCAE
            grid_center_lat = grid_center_lat
            grid_center_lon = grid_center_lon
            grid_width_km = 300
            
            # Create a common grid independent of radar locations
            half_width_km = grid_width_km / 2
            half_width_deg_lat = half_width_km / 111.0
            half_width_deg_lon = half_width_km / (111.0 * np.cos(np.radians(grid_center_lat)))
            
            # Create the common lat/lon grid
            lat_range = np.linspace(grid_center_lat - half_width_deg_lat, 
                                   grid_center_lat + half_width_deg_lat, 
                                   700)
            lon_range = np.linspace(grid_center_lon - half_width_deg_lon, 
                                   grid_center_lon + half_width_deg_lon, 
                                   700)
            grid_lon, grid_lat = np.meshgrid(lon_range, lat_range)
            
            print(f"Created common grid centered at ({grid_center_lat:.4f}, {grid_center_lon:.4f})")
            
            accumulated_mesh = None
            for radar_id in radar_ids:
                conn = nexradaws.NexradAwsInterface()
                t0 = time.time()
                scans = conn.get_avail_scans_in_range(start, end, radar_id)
                print(f"Found {len(scans)} scans for {radar_id}")
                print(f"Time to get available scans: {time.time() - t0:.2f}s")
                
                radar_temp_dir = os.path.join(temp_dir, radar_id)
                os.makedirs(radar_temp_dir, exist_ok=True)
                
                total_files = len(scans)  
                # Download scans
                t0 = time.time()
                results = conn.download(scans, radar_temp_dir)
                print(f"Time to download scans: {time.time() - t0:.2f}s")
                
                # Process each radar volume and accumulate max MESH
                for idx, scan in enumerate(results.iter_success()):
                    print(f"\nProcessing: {radar_id} - {scan.filename} ({idx+1}/{total_files})")
                    if scan.filename.endswith("MDM"):
                        continue
                    print(f"\n=== Processing scan: {scan.filename} ===")
                    print(f"Scan time: {scan.scan_time}")
                    
                    t0 = time.time()
                    radar = scan.open_pyart()
                    print(f"Time to open radar: {time.time() - t0:.2f}s")
                    
                    t0 = time.time()
                    mesh_output = process_radar_volume(radar)
                    print(f"Time to process radar volume: {time.time() - t0:.2f}s")
                    
                    if 'mesh_mh2019_75' not in mesh_output:
                        print("Warning: No MESH data in output!")
                        continue
                    
                    mesh = mesh_output['mesh_mh2019_75']['data']
                    
                    # Extract radar coordinates for processing
                    radar_lat = radar.latitude['data'][0]
                    radar_lon = radar.longitude['data'][0]
                    
                    # For blind cone correction, we need to use the radar's actual location
                    # but still reference our common grid coordinates
                    
                    # Apply appropriate transforms to project the radar data onto the common grid
                    
                    # Create grid coordinates based on this specific radar's location
                    x = np.linspace(-200000, 200000, 700)
                    y = np.linspace(-200000, 200000, 700)
                    X, Y = np.meshgrid(x, y)
                    
                    # Convert Cartesian (X,Y) to lat-lon for this radar
                    radar_grid_lat = radar_lat + (Y / 111000.0)
                    radar_grid_lon = radar_lon + (X / (111000.0 * np.cos(np.radians(radar_lat))))
                    
                    # Now we need to interpolate from radar-specific grid to our common grid
                    from scipy.interpolate import griddata
                    
                    # Flatten the arrays for interpolation
                    points = np.column_stack((radar_grid_lat.flatten(), radar_grid_lon.flatten()))
                    values = mesh[0].flatten()
                    
                    # Mask out NaN values
                    valid = ~np.isnan(values)
                    
                    # Interpolate onto common grid (using nearest for speed, could use linear for more accuracy)
                    if np.any(valid):
                        interpolated_mesh = griddata(
                            points[valid], values[valid], 
                            (grid_lat, grid_lon), 
                            method='nearest', 
                            fill_value=0
                        )
                    else:
                        interpolated_mesh = np.zeros_like(grid_lat)
                    
                    # Update accumulated MESH
                    if accumulated_mesh is None:
                        accumulated_mesh = interpolated_mesh
                    else:
                        # Maximum at each grid cell over time and across radars
                        accumulated_mesh = np.maximum(accumulated_mesh, interpolated_mesh)
                    
                    # Clean up radar object to free memory
                    del radar
                
                # Clean up temp directory
                print("Cleaning up temporary files...")
                for file in os.listdir(radar_temp_dir):
                    os.remove(os.path.join(radar_temp_dir, file))
                os.rmdir(radar_temp_dir)
            
            # Save grid data
            if accumulated_mesh is not None:
                print(f"Saving grid data to {grid_path}")
                np.savez(
                    grid_path,
                    grid_mesh=accumulated_mesh,
                    grid_lat=grid_lat,
                    grid_lon=grid_lon
                )
        
        # Once all scans are processed, compute hail polygons from the accumulated MESH
        if accumulated_mesh is not None:
            print("Computing hail bands from accumulated MESH data")
            bands = get_hail_bands(
                [accumulated_mesh], 
                grid_lat, 
                grid_lon, 
                min_distance_km=min_distance_km,
                output_file=output_file
            )
            return bands
        else:
            print("No MESH data was accumulated. Check scan availability.")
            return {}
            
    except Exception as e:
        import traceback
        print(f"Error processing time range {start} to {end}")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print("\nFull traceback:")
        traceback.print_exc()
        return {}

if __name__ == "__main__":
    # Example usage
    start_time = pd.Timestamp(2023, 5, 9, 7, tz='EST')
    end_time = pd.Timestamp(2023, 5, 9, 10, tz='EST')
    
    hail_bands = main_loop(
        start=start_time,
        end=end_time,
        radar_ids=['KGSP', 'KCAE'],
        temp_dir="./files",
        output_file="hail_contours.txt",
        grid_dir="./grids",
        min_distance_km=7
    )
    
    # Print out the bands and their regions
    if hail_bands:
        for size, info in hail_bands.items():
            print(f"\nHail size: {size} (>= {info['threshold_mm']} mm)")
            for i, polygon in enumerate(info['boundary_points'], start=1):
                print(f" Region {i}: {len(polygon)} boundary points")
                # Example: print first few points
                for pt in polygon[:5]:
                    print(f"  - {pt[0]:.4f}, {pt[1]:.4f}")
                if len(polygon) > 5:
                    print("  ...")
    else:
        print("No hail bands were detected")
