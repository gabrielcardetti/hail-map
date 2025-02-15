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
        (2024, 1, 1),
        (2024, 1, 2),
        (2024, 1, 4),
        (2024, 1, 5),
        (2024, 1, 6),
        (2024, 1, 7),
        (2024, 1, 8),
        (2024, 1, 9),
        (2024, 1, 10),
        (2024, 1, 11),
        (2024, 1, 12),
        (2024, 1, 13),
        (2024, 1, 14),
        (2024, 1, 15),
        (2024, 1, 16),
        (2024, 1, 17),
        (2024, 1, 18),
        (2024, 1, 20),
        (2024, 1, 21),
        (2024, 1, 22),
        (2024, 1, 23),
        (2024, 1, 24),
        (2024, 1, 25),
        (2024, 1, 26),
        (2024, 1, 27),
        (2024, 1, 28),
        (2024, 1, 29),
        (2024, 2, 3),
        (2024, 2, 4),
        (2024, 2, 13),
        (2024, 2, 14),
        (2024, 2, 18),
        (2024, 2, 19),
        (2024, 3, 2),
        (2024, 3, 3),
        (2024, 3, 5),
        (2024, 3, 6),
        (2024, 3, 8),
        (2024, 3, 9),
        (2024, 3, 10),
        (2024, 3, 11),
        (2024, 3, 12),
        (2024, 3, 13),
        (2024, 3, 14),
        (2024, 3, 15),
        (2024, 3, 16),
        (2024, 3, 19),
        (2024, 3, 20),
        (2024, 3, 21),
        (2024, 3, 22),
        (2024, 3, 23),
        (2024, 3, 24),
        (2024, 3, 26),
        (2024, 3, 27),
        (2024, 3, 28),
        (2024, 3, 30),
        (2024, 3, 31),
        (2024, 4, 1),
        (2024, 4, 3),
        (2024, 4, 4),
        (2024, 4, 5),
        (2024, 4, 6),
        (2024, 4, 7),
        (2024, 4, 8),
        (2024, 4, 9),
        (2024, 4, 10),
        (2024, 4, 11),
        (2024, 4, 12),
        (2024, 4, 13),
        (2024, 4, 14),
        (2024, 4, 18),
        (2024, 4, 19),
        (2024, 4, 20),
        (2024, 4, 21),
        (2024, 4, 22),
        (2024, 4, 24),
        (2024, 4, 25),
        (2024, 4, 26),
        (2024, 4, 27),
        (2024, 4, 28),
        (2024, 4, 29),
        (2024, 4, 30),
        (2024, 5, 3),
        (2024, 5, 5),
        (2024, 5, 6),
        (2024, 5, 7),
        (2024, 5, 8),
        (2024, 5, 10),
        (2024, 5, 11),
        (2024, 5, 12),
        (2024, 5, 14),
        (2024, 5, 15),
        (2024, 5, 16),
        (2024, 5, 17),
        (2024, 5, 19),
        (2024, 5, 20),
        (2024, 5, 21),
        (2024, 5, 22),
        (2024, 5, 23),
        (2024, 5, 24),
        (2024, 5, 25),
        (2024, 5, 27),
        (2024, 5, 28),
        (2024, 5, 29),
        (2024, 5, 30),
        (2024, 5, 31),
        (2024, 6, 1),
        (2024, 6, 2),
        (2024, 6, 3),
        (2024, 6, 5),
        (2024, 6, 6),
        (2024, 6, 7),
        (2024, 6, 8),
        (2024, 6, 9),
        (2024, 6, 16),
        (2024, 6, 18),
        (2024, 6, 19),
        (2024, 6, 21),
        (2024, 6, 22),
        (2024, 6, 23),
        (2024, 6, 25),
        (2024, 6, 27),
        (2024, 6, 29),
        (2024, 7, 1),
        (2024, 7, 2),
        (2024, 7, 3),
        (2024, 7, 4),
        (2024, 7, 6),
        (2024, 7, 7),
        (2024, 7, 8),
        (2024, 7, 9),
        (2024, 7, 10),
        (2024, 7, 11),
        (2024, 7, 14),
        (2024, 7, 17),
        (2024, 7, 18),
        (2024, 7, 19),
        (2024, 7, 21),
        (2024, 7, 22),
        (2024, 7, 24),
        (2024, 7, 26),
        (2024, 7, 27),
        (2024, 7, 29),
        (2024, 8, 1),
        (2024, 8, 3),
        (2024, 8, 4),
        (2024, 8, 5),
        (2024, 8, 7),
        (2024, 8, 9),
        (2024, 8, 11),
        (2024, 8, 13),
        (2024, 8, 17),
        (2024, 8, 19),
        (2024, 8, 21),
        (2024, 8, 22),
        (2024, 8, 23),
        (2024, 8, 25),
        (2024, 8, 26),
        (2024, 8, 27),
        (2024, 8, 28),
        (2024, 8, 29),
        (2024, 8, 30),
        (2024, 8, 31),
        (2024, 9, 4),
        (2024, 9, 5),
        (2024, 9, 6),
        (2024, 9, 8),
        (2024, 9, 10),
        (2024, 9, 11),
        (2024, 9, 12),
        (2024, 9, 14),
        (2024, 9, 18),
        (2024, 9, 21),
        (2024, 9, 22),
        (2024, 9, 23),
        (2024, 9, 24),
        (2024, 9, 28),
        (2024, 9, 29),
        (2024, 9, 30),
        (2024, 10, 2),
        (2024, 10, 3),
        (2024, 10, 6),
        (2024, 10, 8),
        (2024, 10, 10),
        (2024, 10, 11),
        (2024, 10, 12),
        (2024, 10, 13),
        (2024, 10, 16),
        (2024, 10, 17),
        (2024, 10, 18),
        (2024, 10, 19),
        (2024, 10, 20),
        (2024, 10, 21),
        (2024, 10, 22),
        (2024, 10, 24),
        (2024, 10, 26),
        (2024, 10, 28),
        (2024, 11, 1)
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
