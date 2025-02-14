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
    harcoded_days_to_skip = [
        (2024, 4, 1),
        (2024, 4, 19),
        (2024, 5, 11),
        (2024, 5, 31),
        (2024, 6, 29),
        (2024, 4, 3),
        (2024, 4, 24),
        (2024, 5, 15),
        (2024, 6, 1),
        (2024, 7, 2),
        (2024, 4, 5),
        (2024, 4, 25),
        (2024, 5, 17),
        (2024, 6, 2),
        (2024, 7, 4),
        (2024, 4, 6),
        (2024, 4, 26),
        (2024, 5, 19),
        (2024, 6, 8),
        (2024, 7, 6),
        (2024, 4, 7),
        (2024, 4, 27),
        (2024, 5, 21),
        (2024, 6, 9),
        (2024, 7, 10),
        (2024, 4, 8),
        (2024, 4, 28),
        (2024, 5, 22),
        (2024, 6, 16),
        (2024, 7, 14),
        (2024, 4, 9),
        (2024, 4, 29),
        (2024, 5, 23),
        (2024, 6, 18),
        (2024, 8, 27),
        (2024, 4, 11),
        (2024, 4, 30),
        (2024, 5, 24),
        (2024, 6, 19),
        (2024, 4, 13),
        (2024, 5, 3),
        (2024, 5, 27),
        (2024, 6, 21),
        (2024, 4, 14),
        (2024, 5, 5),
        (2024, 5, 29),
        (2024, 6, 22),
        (2024, 4, 18),
        (2024, 5, 7),
        (2024, 5, 30),
        (2024, 6, 25)
    ]
    for month in range(1, 13):  # 1 = January, 12 = December
        # Get the number of days in the month
        if month in [4, 6, 9, 11]:  # April, June, September, November have 30 days
            days = 30
        elif month == 2:  # February
            days = 29  # 2024 is a leap year
        else:  # Other months have 31 days
            days = 31

        # Add each day of the month
        for day in range(1, days + 1):
            if (2024, month, day) in harcoded_days_to_skip:
                continue
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
