import numpy as np
import pandas as pd
import pyart
import nexradaws
from scipy import ndimage
from mesh_grid import main as hail_mesh  # Renamed import for clarity
import time

def get_temperature_levels():
    """
    Provide environmental temperature levels for hail algorithm.
    Returns [freezing_level, negative20C_level] in meters.
    """
    # In a real-world scenario, these could be fetched from a sounding or model data.
    freezing_level = 3000   # 0°C level (default 3.2 km, can adjust per environment)
    neg20_level    = 6000   # -20°C level (default 6.5 km)
    return [freezing_level, neg20_level]

def process_radar_volume(radar):
    """
    Grid a single radar volume to Cartesian coordinates and compute hail metrics.
    Returns a dictionary with MESH and related fields.
    """
    # Define grid size and extent
    grid_shape = (20, 500, 500)  # (vertical levels, y, x)
    grid_limits = ((0, 20000),    # 0 to 20 km altitude
                   (-150000, 150000),  # y: -150 to +150 km
                   (-150000, 150000))  # x: -150 to +150 km

    # Use a tighter radius and Cressman weighting to avoid oversmoothing hail cores
    radar_fields = ['reflectivity']
    grid = pyart.map.grid_from_radars(
        (radar,),
        grid_shape=grid_shape,
        grid_limits=grid_limits,
        fields=radar_fields,
        weighting_function="Cressman",
        constant_roi=1000.0,         # 1 km radius of influence for interpolation
        roi_func='constant'          # use constant radius (override default dist_beam)
    )
    # Get environmental levels and compute hail fields (MESH, POSH, etc.)
    levels = get_temperature_levels()
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

def get_hail_bands(mesh_data, lat, lon, min_distance_km=1, output_file=None):
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
    # Hail size thresholds (in mm) corresponding to various size bands
    thresholds = {
        "0.75inch": 19,
        "1inch": 25,
        "1.2inch": 32,
        "1.5inch": 38,
        "1.75inch": 44,
        "2inch": 51,
        "2.5inch": 64,
        "3inch":   76,
        "4inch":   102
    }

    # Modify file opening to use the output_file parameter
    if output_file:
        f = open(output_file, 'w')
    else:
        f = open('hail_contours.txt', 'w')

    try:
        f.write("Hail Contour Data\n")
        f.write("=================\n\n")

        bands = {}
        mesh = mesh_data[0]  # 2D MESH field (assumed at index 0 of mesh_data list)
        grid_res_km = 0.6    # horizontal grid resolution in km (600 m)
        # Compute structuring element size (in pixels) for morphological operations
        structure_size = max(1, int(min_distance_km / grid_res_km))
        structure = np.ones((structure_size, structure_size), dtype=bool)

        # Identify hail regions for each threshold
        for name, threshold in thresholds.items():
            # Create a mask of where MESH exceeds this threshold
            mask = mesh > threshold
            if not np.any(mask):
                continue  # no hail of this size
            # Morphological filtering: close small gaps within `min_distance_km`
            dilated = ndimage.binary_dilation(mask, structure=structure)
            filled = ndimage.binary_erosion(dilated, structure=structure)
            # Label connected hail areas
            labeled_array, num_regions = ndimage.label(filled, structure=np.ones((3,3), dtype=int))
            boundary_points_list = []

            f.write(f"\n{name} (threshold: {threshold}mm)\n")
            f.write("-" * 40 + "\n")
            
            for region_idx in range(1, num_regions + 1):
                region_mask = (labeled_array == region_idx)
                # Skip tiny regions (noise) by area – require at least ~4 pixels (~0.86 km^2)
                if np.sum(region_mask) < 4:
                    continue
                # Find boundary by looking at the gradient (edge) of the region mask
                grad_y, grad_x = np.gradient(region_mask.astype(float))
                boundary_mask = (grad_x**2 + grad_y**2) > 0
                if not np.any(boundary_mask):
                    continue
                y_idx, x_idx = np.where(boundary_mask)
                # Collect lat-lon coordinates of the boundary
                boundary_points = np.column_stack((lat[y_idx, x_idx], lon[y_idx, x_idx]))
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
        return bands

    finally:
        f.close()

def main_loop(
    start: pd.Timestamp = pd.Timestamp(2023, 5, 9, 19, tz='UTC'),
    end: pd.Timestamp = pd.Timestamp(2023, 5, 10, 1, tz='UTC'),
    radar_id: str = 'KCAE',
    temp_dir: str = "./files",
    output_file: str = None
) -> dict:
    """
    Process radar data for a given time range.
    
    Args:
        start: Start time for processing (default: 2023-05-09 19:00 UTC)
        end: End time for processing (default: 2023-05-10 01:00 UTC)
        radar_id: Radar station identifier (default: 'KCAE')
        temp_dir: Directory for temporary files (default: "./files")
        output_file: Path for output file (default: None)
    
    Returns:
        dict: Dictionary containing hail band information
    """
    # Set up AWS NEXRAD data access
    conn = nexradaws.NexradAwsInterface()
    scans = conn.get_avail_scans_in_range(start, end, radar_id)
    print(f"Found {len(scans)} scans for {radar_id}")
    results = conn.download(scans, temp_dir)
    
    # Initialize accumulation grid for MESH
    accumulated_mesh = None
    grid_lat = grid_lon = None

    try:
        # Process each radar volume and accumulate max MESH
        for scan in results.iter_success():
            # Only process base radar files (exclude any intermediate files if present)

            if scan.filename.endswith("MDM"):
                continue
            start_time = time.time()
            print(f"Processing: {scan.filename}")
            radar = scan.open_pyart()  # read the radar volume
            mesh_output = process_radar_volume(radar)
            mesh = mesh_output['mesh_mh2019_75']['data']  # 3D MESH grid (1st level contains 2D field)

            # Extract latitude/longitude grids on first iteration
            if accumulated_mesh is None:
                # Coordinates for the grid based on radar location and x,y offsets
                # Create grid coordinates (only need to do this once)
                x = np.linspace(-150000, 150000, 500)
                y = np.linspace(-150000, 150000, 500)
                X, Y = np.meshgrid(x, y)
                radar_lat = radar.latitude['data'][0]
                radar_lon = radar.longitude['data'][0]
                # Convert Cartesian (X,Y) to lat-lon (approximate, assuming small earth curvature)
                grid_lat = radar_lat + (Y / 111000.0)
                grid_lon = radar_lon + (X / (111000.0 * np.cos(np.radians(radar_lat))))
                # Update accumulated MESH as the maximum at each grid cell over time
                accumulated_mesh = mesh[0]  # initialize with first scan's MESH
            else:
                accumulated_mesh = np.maximum(accumulated_mesh, mesh[0])
            # Clean up radar object to free memory (if needed)
            del radar
            print(f"Finsihed: {scan.filename} on {time.time() - start_time:.2f} seconds")


        # Once all scans are processed, compute hail polygons from the accumulated MESH
        bands = get_hail_bands(
            [accumulated_mesh], 
            grid_lat, 
            grid_lon, 
            min_distance_km=1,
            output_file=output_file
        )
        return bands
    except Exception as e:
        print(f"Error processing time range {start} to {end}: {str(e)}")
        return None

if __name__ == "__main__":
    hail_bands = main_loop()
    for size, info in hail_bands.items():
        print(f"\nHail size: {size} (>= {info['threshold_mm']} mm)")
        for i, polygon in enumerate(info['boundary_points'], start=1):
            print(f" Region {i}: {len(polygon)} boundary points")
            for pt in polygon[:5]:
                print(f"  - {pt[0]:.4f}, {pt[1]:.4f}")
            if len(polygon) > 5:
                print("  ...")