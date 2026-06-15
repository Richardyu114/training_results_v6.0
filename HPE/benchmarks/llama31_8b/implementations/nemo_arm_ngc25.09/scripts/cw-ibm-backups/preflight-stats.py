import sys
import statistics

def read_data(filename):
    names = []
    times = []
    with open(filename, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            names.append(parts[0])
            try:
                times.append(float(parts[1]))
            except ValueError:
                continue
    return names, times

def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <filename> <outlier condition>")
        sys.exit(1)

    names, times = read_data(sys.argv[1])

    if not times:
        print("No valid data found")
        sys.exit(1)

    median_time = statistics.median(times)
    outlier_cond = float(sys.argv[2]) # Outlier condition
    threshold = median_time * outlier_cond
    min_time = min(times)
    avg_time = sum(times)/len(times)
    max_time = max(times)
    stdev_time = statistics.stdev(times) if len(times) > 1 else 0

    print(f"Min:    {min_time:.3f}")
    print(f"Avg:    {avg_time:.3f}")
    print(f"Median: {median_time:.3f}")
    print(f"Max:    {max_time:.3f}")
    print(f"Stdev:  {stdev_time:.3f}")
    print(f"\nOutlier condition: times > {outlier_cond}x median ({threshold:.3f})")

    print("\nOutliers:")
    for name, time in zip(names, times):
        if time > threshold:
            x_factor = time / median_time
            print(f"{name}\t{time:.3f}\t({x_factor:.2f}x median)")

if __name__ == "__main__":
        main()
