import time
from typing import List

import numpy as np


def match_timestamps(timestamps1: np.ndarray, timestamps2: np.ndarray) -> np.ndarray:
    matching = np.zeros(len(timestamps1), dtype=int)

    # Initialize a variable `j to keep track of the closest timestamp in timestamps2
    j = 0

    for i, t1 in enumerate(timestamps1):
        # Calculate the absolute difference for current index j
        current_diff = abs(timestamps2[j] - t1)

        # While the difference for the next index in timestamps2 is smaller, move `j forward
        while j + 1 < len(timestamps2) and abs(timestamps2[j + 1] - t1) < current_diff:
            j += 1
            current_diff = abs(timestamps2[j] - t1)

        # Store the index of the closest timestamp in matching array
        matching[i] = j

    return matching


def make_timestamps(fps: int, st_ts: float, fn_ts: float) -> np.ndarray:
    # Calculate the total number of frames based on fps and duration
    num_frames = int((fn_ts - st_ts) * fps)

    # Generate uniform timestamps between start and end timestamps
    timestamps = np.linspace(st_ts, fn_ts, num_frames, endpoint=False)

    # Add random noise to create variability in timestamp spacing
    noise = np.random.randn(len(timestamps)) * (1.0 / fps * 0.1)  # Adjust noise magnitude if needed
    timestamps += noise

    # Ensure timestamps are sorted and have unique values
    timestamps = np.unique(np.sort(timestamps))

    return timestamps


def main():
    """
       Setup and match timestamps between two cameras.
       """
    # Generate timestamps for the first camera
    timestamps1 = make_timestamps(30, time.time() - 100, time.time() + 3600 * 2)    # np.array([0, 0.091, 0.5])
    # Generate timestamps for the second camera
    timestamps2 = make_timestamps(60, time.time() + 200, time.time() + 3600 * 2.5)  # np.array([0.001, 0.09, 0.12, 0.6])

    start = time.time()
    # Find matching timestamps
    matching = match_timestamps(timestamps1, timestamps2)
    end = time.time()
    print(f"Number of matches found: {len(matching)}")
    if len(matching) > 0:
        print("Sample matches:", matching)
    print(f"Time: {end - start}")



if __name__ == '__main__':
    main()