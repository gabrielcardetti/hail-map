import numpy as np
import pandas as pd
import pyart
import nexradaws
from scipy import ndimage
from mesh_ppi import main as hail_mesh  # Changed from mesh_grid to mesh_ppi
from scipy.interpolate import griddata
import os
from dataclasses import dataclass
import time

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
    freezing_level = 3000   # 0°C level (default 3.2 km, can adjust per environment)
    neg20_level    = 6000   # -20°C level (default 6.5 km)
    return [freezing_level, neg20_level]

def process_radar_volume(radar):
    # Get environmental levels
    levels = get_temperature_levels()
    # Apply MESH calculation directly on PPI data
    mesh_data = hail_mesh(
        radar,
        'reflectivity',
        levels,
    )
    if mesh_data is None:
        return None
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

def get_hail_bands(mesh_data, lat, lon, min_distance_km=2, output_file=None):
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

@dataclass
class LocalScan:
    filename: str
    filepath: str
    
    def open_pyart(self):
        return pyart.io.read(self.filepath)

def get_local_scans(scans, temp_dir: str) -> list[LocalScan]:
    local_scans = []
    scans_to_download = []
    
    # Check which files need to be downloaded
    for scan in scans:
        local_path = os.path.join(temp_dir, scan.filename)
        if os.path.exists(local_path):
            local_scans.append(LocalScan(scan.filename, local_path))
        else:
            scans_to_download.append(scan)
    
    print(f"Found {len(local_scans)} existing files")
    if scans_to_download:
        print(f"Downloading {len(scans_to_download)} new files")
        conn = nexradaws.NexradAwsInterface()
        results = conn.download(scans_to_download, temp_dir)
        local_scans.extend([LocalScan(scan.filename, scan.filepath) for scan in results.iter_success()])
    
    return local_scans

def main_loop(
    start: pd.Timestamp = pd.Timestamp(2023, 5, 9, 17, tz='EST'),
    end: pd.Timestamp = pd.Timestamp(2023, 5, 9, 21, tz='EST'),
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
    # Set up AWS NEXRAD data access and get available scans
    conn = nexradaws.NexradAwsInterface()
    scans = conn.get_avail_scans_in_range(start, end, radar_id)
    print(f"Found {len(scans)} scans for {radar_id}")
    
    # Get local scans, downloading if necessary
    local_scans = get_local_scans(scans, temp_dir)
    
    # Initialize the accumulation grid on first scan
    grid_mesh = None
    grid_lat = grid_lon = None
    
    try:
        # Process each radar volume and accumulate max MESH
        total_files = len(local_scans)
        for idx, scan in enumerate(local_scans, 1):
            if scan.filename.endswith("MDM"):
                continue
                
            start_time = time.time()
            print(f"Processing: {scan.filename} ({idx}/{total_files})")
            radar = scan.open_pyart()
            scan_mesh_points = process_radar_volume(radar)
            if scan_mesh_points is None:
                continue
            
            # Get points from this scan
            scan_lats = np.array([p['lat'] for p in scan_mesh_points])
            scan_lons = np.array([p['lon'] for p in scan_mesh_points])
            scan_mesh = np.array([p['mesh_value'] for p in scan_mesh_points])
            
            # Create or reuse grid
            if grid_mesh is None:
                # Initialize grid on first scan
                lat_min, lat_max = np.min(scan_lats), np.max(scan_lats)
                lon_min, lon_max = np.min(scan_lons), np.max(scan_lons)
                grid_lat, grid_lon = np.mgrid[lat_min:lat_max:500j, lon_min:lon_max:500j]
                grid_mesh = np.zeros_like(grid_lat)
            
            # Interpolate this scan's MESH values to grid
            scan_grid = griddata(
                (scan_lats, scan_lons),
                scan_mesh,
                (grid_lat, grid_lon),
                method='nearest',
                fill_value=0
            )
            
            # Update accumulation grid with maximum values
            grid_mesh = np.maximum(grid_mesh, scan_grid)
            
            print(f"Finsihed: {scan.filename} on {time.time() - start_time:.2f} seconds")
            # print("-" * 40)
            
            del radar, scan_grid
            
        if grid_mesh is not None:
            # Process accumulated MESH into hail bands
            bands = get_hail_bands(
                [grid_mesh],
                grid_lat,
                grid_lon,
                min_distance_km=2,
                output_file=output_file
            )
            return bands
            
    except Exception as e:
        import traceback
        print(f"Error processing time range {start} to {end}")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print("\nFull traceback:")
        traceback.print_exc()
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