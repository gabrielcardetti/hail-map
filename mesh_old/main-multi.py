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
    radar_ids = ['KGSP', 'KCAE']

    # Construct output filename for hail contours
    output_dir = "./contours-kgsp-kcae-2025-until-march-11"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(
        output_dir,
        f"hail_contours_multiple_{start_time.strftime('%Y%m%d_%H%M')}.txt"
    )

    # Create unique directory for each time range to store temporary radar files
    temp_dir = f"./files/file_{radar_ids}_{start_time.strftime('%Y%m%d_%H%M')}"

    print(f"\nProcessing period: {start_time} to {end_time}")

    try:
        # Call main processing function with all necessary parameters
        result = main_loop(
            start=start_time,
            end=end_time,
            radar_ids=radar_ids,
            temp_dir=temp_dir,
            output_file=output_file
        )
        total_range_time = time.time() - range_start_time
        print(
            f"✓ Successfully processed period: {time_range[0]} to {time_range[1]} it takes {total_range_time:.2f} seconds")

        # empty the temp dir
        for file in os.listdir(temp_dir):
            os.remove(os.path.join(temp_dir, file))
        os.rmdir(temp_dir)
        print(f"✓ Successfully emptied temp dir: {temp_dir}")
        return result
    except TypeError as e:
        print(f"Error processing time range {start_time} to {end_time}")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        return None


if __name__ == "__main__":
    time_ranges = []
    
    # Process all days for 2022, 2023, and 2024
    for year in [2022, 2023, 2024]:
        for month in range(1, 13):  # All 12 months
            # Determine days in month (accounting for leap years)
            if month in [4, 6, 9, 11]:
                days = 30
            elif month == 2:
                # Check for leap year
                if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
                    days = 29
                else:
                    days = 28
            else:
                days = 31
                
            # Add each day of the month
            for day in range(1, days + 1):
                time_ranges.append(
                    (
                        pd.Timestamp(year, month, day, 0, tz='UTC'),
                        pd.Timestamp(year, month, day, 23, 59, tz='UTC')
                    )
                )
    
    # Process January and partial February 2025 (keeping existing code)
    year = 2025
    for month in [1, 2, 3]:  # January and February only
        if month == 1:  # January
            days = 31
        elif month == 2:  # February
            days = 28  # Only process until Feb 17
        else:  # March
            days = 9

        # Add each day of the month
        for day in range(1, days + 1):
            time_ranges.append(
                (
                    pd.Timestamp(year, month, day, 0, tz='UTC'),
                    pd.Timestamp(year, month, day, 23, 59, tz='UTC')
                )
            )

    num_cores = max(1, mp.cpu_count() - 15)

    # num_cores = 1
    print(f"Processing using {num_cores} CPU cores")

    # Create a pool of workers
    with mp.Pool(processes=num_cores) as pool:
        # Process time ranges in parallel
        results = pool.map(process_time_range, time_ranges)

    # Process results
    for time_range, result in zip(time_ranges, results):
        if result is None:
            print(
                f"✗ Failed to process period: {time_range[0]} to {time_range[1]}")
