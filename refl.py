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

def get_grid_filename(start, end, radar_ids):
    """Generate a unique filename for the grid data based on parameters"""
    start_str = start.strftime("%Y%m%d%H")
    end_str = end.strftime("%Y%m%d%H")
    radar_str = "_".join(radar_ids)
    return f"refl_grid_{start_str}_{end_str}_{radar_str}.npz"

def main_loop(
    start: pd.Timestamp = pd.Timestamp(2023, 8, 7, 13, tz='EST'),
    end: pd.Timestamp = pd.Timestamp(2023, 8, 7, 19, tz='EST'),
    radar_ids: List[str] = ['KGSP'],
    temp_dir: str = "./files",
    output_file: str = None,
    grid_dir: str = "./grids",
    smooth_sigma: float = 1.0,
    output_plot: str = None,
    generate_bands: bool = True,
    plot_only: bool = False,
    plot_refl: bool = False,
    refl_plot_dir: str = "./reflectivity_plots"
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
        generate_bands: Whether to generate hail bands or just plot the data
        plot_only: If True, only plot existing grid data without processing radars
        plot_refl: Whether to plot base reflectivity for each volume
        refl_plot_dir: Directory to save reflectivity plots if plot_refl is True
    """
    # Create necessary directories
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(grid_dir, exist_ok=True)
    
    # Generate grid filename for saving/loading
    grid_filename = get_grid_filename(start, end, radar_ids)
    grid_filepath = os.path.join(grid_dir, grid_filename)
    
    # Check if we can load existing grid data
    if plot_only or os.path.exists(grid_filepath):
        if os.path.exists(grid_filepath):
            print(f"Loading existing grid data from {grid_filepath}")
            grid_data = np.load(grid_filepath)
            grid_mesh = grid_data['mesh']
            grid_lat = grid_data['lat']
            grid_lon = grid_data['lon']
        else:
            print("Error: Plot-only mode requested but grid file not found.")
            return None
    else:
        # Initialize grid
        grid_mesh = None
        grid_lat = grid_lon = None
        grid_bounds = None
        
        # Connect to NEXRAD AWS interface
        conn = nexradaws.NexradAwsInterface()
        
        # Process each radar
        for radar_id in radar_ids:
            print(f"Processing radar: {radar_id}")
            
            # Get available scans
            scans = conn.get_avail_scans_in_range(start, end, radar_id)
            print(f"Found {len(scans)} scans for {radar_id}")
            
            # Create radar temp directory
            radar_temp_dir = os.path.join(temp_dir, radar_id)
            os.makedirs(radar_temp_dir, exist_ok=True)
            
            # Get local scans
            local_scans = get_local_scans(scans, radar_temp_dir)
            
            # Process each scan
            for scan in local_scans:
                if scan.filename.endswith("MDM"):
                    continue
                    
                print(f"Processing scan: {scan.filename}")
                
                # Open radar data
                radar = scan.open_pyart()
                
                # Extract reflectivity data and coordinates
                sweep = 0  # Use the lowest sweep (base reflectivity)
                refl_data = radar.get_field(sweep, 'reflectivity', copy=True).filled(np.nan)
                
                # Get lat/lon coordinates for each point
                azimuth = radar.get_azimuth(sweep)
                ranges = radar.range['data']
                
                # Filter out points beyond 150km
                max_range = 150000  # 150km in meters
                range_mask = ranges > max_range
                range_grid = np.tile(range_mask, (len(azimuth), 1))
                refl_data[range_grid] = 0
                
                # Get radar location
                radar_lat = radar.latitude['data'][0]
                radar_lon = radar.longitude['data'][0]
                
                # Convert polar coordinates to Cartesian
                az_rad = np.deg2rad(azimuth)
                x = ranges * np.sin(az_rad[:, np.newaxis])
                y = ranges * np.cos(az_rad[:, np.newaxis])
                
                # Convert to lat/lon
                R = 6371000  # Earth radius in meters
                scan_lats = radar_lat + (y / R) * (180 / np.pi)
                scan_lons = radar_lon + (x / R) * (180 / np.pi) / np.cos(np.deg2rad(radar_lat))
                
                # Update grid bounds
                if grid_bounds is None:
                    lat_min, lat_max = np.nanmin(scan_lats), np.nanmax(scan_lats)
                    lon_min, lon_max = np.nanmin(scan_lons), np.nanmax(scan_lons)
                    grid_bounds = (lat_min, lat_max, lon_min, lon_max)
                else:
                    lat_min, lat_max = np.nanmin(scan_lats), np.nanmax(scan_lats)
                    lon_min, lon_max = np.nanmin(scan_lons), np.nanmax(scan_lons)
                    grid_bounds = (
                        min(grid_bounds[0], lat_min),
                        max(grid_bounds[1], lat_max),
                        min(grid_bounds[2], lon_min),
                        max(grid_bounds[3], lon_max)
                    )
                
                # Create the grid if it doesn't exist
                if grid_mesh is None:
                    grid_lat, grid_lon = np.mgrid[
                        grid_bounds[0]:grid_bounds[1]:500j,
                        grid_bounds[2]:grid_bounds[3]:500j
                    ]
                    grid_mesh = np.zeros_like(grid_lat)
                
                # Flatten data for griddata
                valid_points = ~np.isnan(refl_data)
                points = np.column_stack((
                    scan_lats[valid_points],
                    scan_lons[valid_points]
                ))
                values = refl_data[valid_points]
                
                if len(points) > 0:
                    # Interpolate reflectivity to grid
                    scan_grid = griddata(
                        points,
                        values,
                        (grid_lat, grid_lon),
                        method='linear',
                        fill_value=0
                    )
                    
                    # Update grid with maximum values
                    grid_mesh = np.maximum(grid_mesh, scan_grid)

        # Save the grid data for future use
        print(f"Saving grid data to {grid_filepath}")
        np.savez(
            grid_filepath,
            mesh=grid_mesh,
            lat=grid_lat,
            lon=grid_lon
        )
    
    # Subtract 10 from the grid values before plotting
    plot_mesh = grid_mesh - 10
    
    # Plot the resulting grid
    plt.figure(figsize=(12, 10))
    
    # Create custom colormap: black until 50 dBZ, then yellow->red->black
    cmap_colors = []
    
    # Add black for values less than 50 dBZ
    for i in range(0, 55):
        cmap_colors.append((1, 1, 1, 1))  # Black
    
    # Add yellow to red to black gradient for values 50-70 dBZ
    yellow = np.array([1, 1, 0, 1])       # Yellow
    red = np.array([1, 0, 0, 1])          # Red
    black = np.array([0, 0, 0, 1])        # Black
    
    # Yellow to red (50-60 dBZ)
    for i in range(10):
        ratio = i / 10
        color = yellow * (1 - ratio) + red * ratio
        cmap_colors.append(tuple(color))
    
    # Red to black (60-70 dBZ)
    for i in range(10):
        ratio = i / 10
        color = red * (1 - ratio) + black * ratio
        cmap_colors.append(tuple(color))
    
    custom_cmap = mcolors.ListedColormap(cmap_colors)
    
    # Create levels and norm for the colormap
    levels = np.arange(-10, 71, 1)
    norm = BoundaryNorm(levels, len(levels))
    
    # Plot the grid data with 10 subtracted
    mesh_plot = plt.pcolormesh(grid_lon, grid_lat, plot_mesh, cmap=custom_cmap, norm=norm)
    
    # Add colorbar (showing the original values by adding 10 back to the labels)
    cbar = plt.colorbar(mesh_plot, label='Reflectivity (dBZ)')
    cbar.set_ticks(np.arange(0, 71, 10))
    cbar.set_ticklabels(np.arange(10, 81, 10))
    
    # Set labels and title
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title(f'Radar Reflectivity Grid\n{", ".join(radar_ids)} - {start} to {end}')
    plt.grid(True, alpha=0.3)
    
    # Save or show the plot
    if output_plot:
        plt.savefig(output_plot, dpi=300, bbox_inches='tight')
        print(f"Grid plot saved to {output_plot}")
    else:
        plt.show()
    
    # Return the grid data
    return {
        "mesh": grid_mesh,
        "lat": grid_lat,
        "lon": grid_lon
    }

if __name__ == "__main__":
    # Example usage with reflectivity plotting and mesh output
    result = main_loop(
        plot_refl=True,
        refl_plot_dir="./reflectivity_plots",
        output_plot="mesh_output.png"
    )