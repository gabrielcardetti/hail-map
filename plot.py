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
from typing import List
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import BoundaryNorm

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
    # Quick check for potential hail before full processing
    max_refl = -999
    for sweep_idx in range(radar.nsweeps):
        refl_data = radar.get_field(sweep_idx, 'reflectivity', copy=True).filled(np.nan)
        curr_max = np.nanmax(refl_data)
        if curr_max > max_refl:
            max_refl = curr_max
    
    if max_refl < 65:
        print(f"Skipping volume - maximum reflectivity ({max_refl:.1f} dBZ) below hail threshold")
        return None
        
    levels = get_temperature_levels()
    mesh_data = hail_mesh(
        radar,
        'reflectivity',
        levels,
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

def thresholded_hail_correction(n):
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

    radar_diff = 17
    corrected = np.where(n < 40, np.maximum(n - radar_diff, 0), n)
    mask = radar_diff * (1 - (n - 40) / 50)
    corrected = np.where((n >= 40) & (n < 90), n - mask, corrected)

    return corrected

def smooth_mesh_data(mesh_data, sigma=1.0):
    """
    Apply Gaussian smoothing to the mesh data to create separation between hail bands.
    
    Args:
        mesh_data: 2D grid of MESH values
        sigma: Standard deviation for Gaussian kernel
        
    Returns:
        Smoothed mesh data
    """
    from scipy.ndimage import gaussian_filter
    
    # Make a copy to avoid modifying the original
    smoothed = mesh_data.copy()
    
    # Apply Gaussian smoothing
    smoothed = gaussian_filter(smoothed, sigma=sigma)
    
    return smoothed

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
            mask = thresholded_hail_correction(mesh) > threshold
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

def get_grid_bounds(scan_lats: np.ndarray, scan_lons: np.ndarray, grid_bounds=None):
    """
    Calculate or update grid bounds based on scan data.
    
    Args:
        scan_lats: Latitude points from current scan
        scan_lons: Longitude points from current scan
        grid_bounds: Optional existing bounds (min_lat, max_lat, min_lon, max_lon)
    
    Returns:
        Tuple of (min_lat, max_lat, min_lon, max_lon)
    """
    lat_min, lat_max = np.min(scan_lats), np.max(scan_lats)
    lon_min, lon_max = np.min(scan_lons), np.max(scan_lons)
    
    if grid_bounds is None:
        return (lat_min, lat_max, lon_min, lon_max)
        
    return (
        min(grid_bounds[0], lat_min),
        max(grid_bounds[1], lat_max),
        min(grid_bounds[2], lon_min),
        max(grid_bounds[3], lon_max)
    )

def process_radar_grid(
    start: pd.Timestamp,
    end: pd.Timestamp,
    radar_ids: List[str],
    temp_dir: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Process radar data and create accumulated grid.
    
    Args:
        start: Start time for processing
        end: End time for processing
        radar_ids: List of radar station identifiers
        temp_dir: Directory for temporary files
    
    Returns:
        Tuple of (grid_mesh, grid_lat, grid_lon)
    """
    conn = nexradaws.NexradAwsInterface()
    grid_mesh = None
    grid_lat = grid_lon = None
    grid_bounds = None
    
    # Process each radar
    for radar_id in radar_ids:
        loop_start = time.time()
        
        # Create radar-specific temp directory
        radar_temp_dir = os.path.join(temp_dir, radar_id)
        os.makedirs(radar_temp_dir, exist_ok=True)
        
        # Get available scans for this radar
        t0 = time.time()
        scans = conn.get_avail_scans_in_range(start, end, radar_id)
        print(f"Found {len(scans)} scans for {radar_id}")
        print(f"Time to get available scans: {time.time() - t0:.2f}s")
        
        # Get local scans, downloading if necessary
        t0 = time.time()
        local_scans = get_local_scans(scans, radar_temp_dir)
        print(f"Time to get/download local scans: {time.time() - t0:.2f}s")
        
        # Process each radar volume and accumulate max MESH
        total_files = len(local_scans)
        for idx, scan in enumerate(local_scans, 1):
            if scan.filename.endswith("MDM"):
                continue
                
            scan_start = time.time()
            print(f"\nProcessing: {radar_id} - {scan.filename} ({idx}/{total_files})")
            
            # Open radar file
            t0 = time.time()
            radar = scan.open_pyart()
            print(f"Time to open radar file: {time.time() - t0:.2f}s")
            
            # Process radar volume
            t0 = time.time()
            scan_mesh_points = process_radar_volume(radar)
            print(f"Time to process radar volume: {time.time() - t0:.2f}s")
            
            if scan_mesh_points is None:
                continue
            
            # Extract points data
            t0 = time.time()
            scan_lats = np.array([p['lat'] for p in scan_mesh_points])
            scan_lons = np.array([p['lon'] for p in scan_mesh_points])
            scan_mesh = np.array([p['mesh_value'] for p in scan_mesh_points])
            
            # Update grid bounds
            t0 = time.time()
            grid_bounds = get_grid_bounds(scan_lats, scan_lons, grid_bounds)
            
            # Create or update grid
            if grid_mesh is None:
                grid_lat, grid_lon = np.mgrid[
                    grid_bounds[0]:grid_bounds[1]:500j,
                    grid_bounds[2]:grid_bounds[3]:500j
                ]
                grid_mesh = np.zeros_like(grid_lat)
            
            # Interpolate to grid
            t0 = time.time()
            scan_grid = griddata(
                (scan_lats, scan_lons),
                scan_mesh,
                (grid_lat, grid_lon),
                method='linear',
                fill_value=0
            )
            print(f"Time to interpolate to grid: {time.time() - t0:.2f}s")
            
            # Update accumulation
            grid_mesh = np.maximum(grid_mesh, scan_grid)
            print(f"Total scan processing time: {time.time() - scan_start:.2f}s")
            
            del radar, scan_grid
        
        # Clean up temp directory
        t0 = time.time()
        for file in os.listdir(radar_temp_dir):
            os.remove(os.path.join(radar_temp_dir, file))
        os.rmdir(radar_temp_dir)
        print(f"\nTime to clean up temp directory: {time.time() - t0:.2f}s")
        print(f"Total radar processing time: {time.time() - loop_start:.2f}s")
    
    return grid_mesh, grid_lat, grid_lon

def get_grid_filename(start: pd.Timestamp, end: pd.Timestamp, radar_ids: List[str]) -> str:
    """Generate a unique filename for the grid data."""
    radar_str = '_'.join(radar_ids)
    return f"grid_{radar_str}_{start.strftime('%Y%m%d_%H%M')}_{end.strftime('%Y%m%d_%H%M')}.npz"

def main_loop(
    start: pd.Timestamp = pd.Timestamp(2023, 8, 7, 13, tz='EST'),
    end: pd.Timestamp = pd.Timestamp(2023, 8, 7, 23, tz='EST'),
    radar_ids: List[str] = ['KGSP', 'KCAE'],
    temp_dir: str = "./files",
    output_file: str = None,
    grid_dir: str = "./grids",
    smooth_sigma: float = 1.0,
    output_plot: str = None
) -> dict:
    """
    Process radar data for multiple radars over a given time range.
    
    Args:
        start: Start time for processing
        end: End time for processing
        radar_ids: List of radar station identifiers
        temp_dir: Directory for temporary files
        output_file: Path for output file
        grid_dir: Directory for saved grid files
        smooth_sigma: Smoothing parameter for mesh data (higher = more smoothing)
        output_plot: Path to save the plot image (if None, will display the plot)
    """
    os.makedirs(grid_dir, exist_ok=True)
    grid_filename = get_grid_filename(start, end, radar_ids)
    grid_path = os.path.join(grid_dir, grid_filename)

    try:
        # Check if grid data exists
        if os.path.exists(grid_path):
            print(f"Loading existing grid data from {grid_path}")
            grid_data = np.load(grid_path)
            grid_mesh = grid_data['grid_mesh']
            grid_lat = grid_data['grid_lat']
            grid_lon = grid_data['grid_lon']
        else:
            # Process radar data to create grid
            grid_mesh, grid_lat, grid_lon = process_radar_grid(start, end, radar_ids, temp_dir)
            
            if grid_mesh is not None:
                # Save grid data
                print(f"Saving grid data to {grid_path}")
                np.savez(
                    grid_path,
                    grid_mesh=grid_mesh,
                    grid_lat=grid_lat,
                    grid_lon=grid_lon
                )
        
        if grid_mesh is not None:
            # Apply smoothing to the mesh data before processing hail bands
            print(f"Applying smoothing with sigma={smooth_sigma} to mesh data")
            smoothed_mesh = smooth_mesh_data(grid_mesh, sigma=smooth_sigma)
            
            # Plot the smoothed mesh data instead of getting hail bands
            print("Plotting smoothed MESH data")
            
            # Create a colormap for MESH values with appropriate levels
            levels = [0, 19, 25, 32, 38, 44, 51, 57, 64, 76, 102, 120]
            colors = ['#FFFFFF', '#98FB98', '#00FF00', '#FFFF00', '#FFD700', 
                     '#FFA500', '#FF4500', '#FF0000', '#B22222', '#8B0000', '#800080']
            cmap = mcolors.ListedColormap(colors)
            norm = BoundaryNorm(levels, cmap.N)
            
            # Create proper 2D meshgrid for plotting
            plt.figure(figsize=(12, 10))
            
            # Transpose both the coordinates and data for correct orientation
            mesh_plot = plt.pcolormesh(grid_lat.T, grid_lon.T, 
                                      thresholded_hail_correction(smoothed_mesh).T, 
                                      cmap=cmap, norm=norm)
            
            # Add colorbar
            cbar = plt.colorbar(mesh_plot, label='Maximum Estimated Size of Hail (mm)')
            
            # Add threshold markers to colorbar
            cbar_ticks = levels
            cbar.set_ticks(cbar_ticks)
            
            # Add size labels to specific ticks
            size_labels = {
                19: "0.75\"", 25: "1\"", 32: "1.2\"", 38: "1.5\"", 
                44: "1.75\"", 51: "2\"", 57: "2.25\"", 64: "2.5\"",
                76: "3\"", 102: "4\""
            }
            cbar_ticklabels = [f"{t} {size_labels.get(t, '')}" for t in cbar_ticks]
            cbar.set_ticklabels(cbar_ticklabels)
            
            # Add grid and set labels
            plt.grid(True, alpha=0.5)
            plt.xlabel('Latitude')
            plt.ylabel('Longitude')
            
            # Set title with time range and radar IDs
            radar_str = ', '.join(radar_ids)
            time_str = f"{start.strftime('%Y-%m-%d %H:%M')} to {end.strftime('%Y-%m-%d %H:%M')} {start.tz}"
            plt.title(f"Maximum Estimated Size of Hail (MESH)\n{radar_str} - {time_str}")
            
            # Save or show the plot
            if output_plot:
                plt.savefig(output_plot, dpi=300, bbox_inches='tight')
                print(f"Plot saved to {output_plot}")
            else:
                plt.show()
                
            return {'plot_created': True}
            
    except Exception as e:
        import traceback
        print(f"Error processing time range {start} to {end}")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print("\nFull traceback:")
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = main_loop(output_plot="mesh_plot.png")
    print(f"Processing complete: {result}")