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
    radar_id = 'KGSP'

    # Construct output filename for hail contours
    output_dir = "./contours-kgsp"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(
        output_dir,
        f"hail_contours_{radar_id}_{start_time.strftime('%Y%m%d_%H%M')}.txt"
    )

    # Create unique directory for each time range to store temporary radar files
    temp_dir = f"./files/file_{radar_id}_{start_time.strftime('%Y%m%d_%H%M')}"

    print(f"\nProcessing period: {start_time} to {end_time}")

    # Call main processing function with all necessary parameters
    result = main_loop(
        start=start_time,
        end=end_time,
        radar_id=radar_id,
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


if __name__ == "__main__":
    time_ranges = []
    for month in range(4, 6):  # 1 = January, 12 = December
        # Get the number of days in the month
        if month in [4, 6, 9, 11]:  # April, June, September, November have 30 days
            days = 30
        elif month == 2:  # February
            days = 29  # 2024 is a leap year
        else:  # Other months have 31 days
            days = 31

        # Add each day of the month
        for day in range(1, days + 1):
            time_ranges.append(
                (
                    pd.Timestamp(2024, month, day, 0, tz='UTC'),
                    pd.Timestamp(2024, month, day, 23, 59, tz='UTC')
                )
            )

    num_cores = max(1, mp.cpu_count() - 3)
    # num_cores = 2
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
