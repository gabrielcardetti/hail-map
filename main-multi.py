import pandas as pd
import multiprocessing as mp
from main import (
    main_loop
)
import os
import time

def process_time_range(time_range: tuple[pd.Timestamp, pd.Timestamp]) -> dict:
    """
    Process radar data for a given time range.
    
    Args:
        time_range (tuple): Tuple containing (start_time, end_time) as pd.Timestamp objects
    """
    range_start_time = time.time()
    start_time, end_time = time_range
    radar_id = 'KCAE'
    
    # Construct output filename for hail contours
    output_dir = "./contours"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(
        output_dir,
        f"hail_contours_{radar_id}_{start_time.strftime('%Y%m%d_%H%M')}.txt"
    )
    
    # Create unique directory for each time range to store temporary radar files
    temp_dir = f"./files/file_{radar_id}_{start_time.strftime('%Y%m%d_%H%M')}"
    
    print(f"\nProcessing period: {start_time} to {end_time}")
    
    # Call main processing function with all necessary parameters
    result =  main_loop(
        start=start_time,
        end=end_time,
        radar_id=radar_id,
        temp_dir=temp_dir,
        output_file=output_file
    )
    total_range_time = time.time() - range_start_time
    print(f"✓ Successfully processed period: {time_range[0]} to {time_range[1]} it takes {total_range_time:.2f} seconds")
    return result

if __name__ == "__main__":
    # Define the time ranges to process
    time_ranges = [
        # Each tuple contains (start_time, end_time) for a processing period
        (pd.Timestamp(2023, 5, 9, 19, 0, tz='UTC'), pd.Timestamp(2023, 5, 9, 21, 59, tz='UTC')),
        (pd.Timestamp(2023, 5, 9, 22, 0, tz='UTC'), pd.Timestamp(2023, 5, 9, 23, 0, tz='UTC')),
    ]

    # time_ranges = [
    #     # May 2024
    #     (pd.Timestamp(2024, 5, 8, 0, tz='UTC'), pd.Timestamp(2024, 5, 8, 23, 59, tz='UTC')),
    #     (pd.Timestamp(2024, 5, 26, 0, tz='UTC'), pd.Timestamp(2024, 5, 26, 23, 59, tz='UTC')),
        
    #     # May 2024
    #     (pd.Timestamp(2024, 5, 6, 0, tz='UTC'), pd.Timestamp(2024, 5, 6, 23, 59, tz='UTC')),
        
    #     # September 2024
    #     (pd.Timestamp(2024, 9, 24, 0, tz='UTC'), pd.Timestamp(2024, 9, 24, 23, 59, tz='UTC')),
        
    #     # August 2024
    #     (pd.Timestamp(2024, 8, 31, 0, tz='UTC'), pd.Timestamp(2024, 8, 31, 23, 59, tz='UTC')),
        
    #     # June 2023
    #     (pd.Timestamp(2023, 6, 6, 0, tz='UTC'), pd.Timestamp(2023, 6, 6, 23, 59, tz='UTC')),
        
    #     # May 2023
    #     (pd.Timestamp(2023, 5, 9, 0, tz='UTC'), pd.Timestamp(2023, 5, 9, 23, 59, tz='UTC')),
        
    #     # May 2021
    #     (pd.Timestamp(2021, 5, 3, 0, tz='UTC'), pd.Timestamp(2021, 5, 3, 23, 59, tz='UTC')),
    # ]

    
    num_cores = max(1, mp.cpu_count() - 1)
    # num_cores = 2
    print(f"Processing using {num_cores} CPU cores")
    
    # Create a pool of workers
    with mp.Pool(processes=num_cores) as pool:
        # Process time ranges in parallel
        results = pool.map(process_time_range, time_ranges)

    # Process results
    for time_range, result in zip(time_ranges, results):
        if result is None:
            print(f"✗ Failed to process period: {time_range[0]} to {time_range[1]}")