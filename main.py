import time
import numpy as np


def match_timestamps(timestamps1: np.ndarray, timestamps2: np.ndarray) -> np.ndarray:

    matching = np.zeros(len(timestamps1), dtype=int)
    j = 0

    for i, t1 in enumerate(timestamps1):
        # Move j to the closest timestamp in timestamps2 for the current timestamp t1
        while j < len(timestamps2) - 1 and abs(timestamps2[j + 1] - t1) < abs(timestamps2[j] - t1):
            j += 1
        # j is now at the best match index in timestamps2 for timestamps1[i]
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
    timestamps1 = make_timestamps(30, time.time() - 100, time.time() + 3600 * 2)
    # Generate timestamps for the second camera
    timestamps2 = make_timestamps(60, time.time() + 200, time.time() + 3600 * 2.5)

    start = time.time()
    # Find matching timestamps
    matching = match_timestamps(timestamps1, timestamps2)
    end = time.time()
    print(f"Number of matches found: {len(matching)}")
    if len(matching) > 0:
        print("Sample matches:", matching[:10])  # Print first 10 matches as a sample
    print(f"Time: {end - start}")


if __name__ == '__main__':
    main()