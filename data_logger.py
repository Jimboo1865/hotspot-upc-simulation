import time
import numpy as np

class PreformanceLogger:
    def __init__(self):
        self.execution_times =[]
        self.hotspot_counts = []

    def log_events(self, start_time: float, end_time: float, num_hotspots: int):
        duration = (end_time - start_time) * 1000.0
        self.execution_times.append(duration)
        self.hotspot_counts.append(num_hotspots)

    def print_summary(self):
        avg_time = np.mean(self.execution_times)
        std_time = np.std(self.execution_times)
        avg_hs = np.mean(self.hotspot_counts)

        print("\n" + "="*45)
        print("         PERFORMANCE PROFILE SUMMARY         ")
        print("="*45)
        print(f"Total Iterations Processed : {len(self.execution_times)}")
        print(f"Avg Hot Spots per Nucleus  : {avg_hs / 208.0:.2f}")
        print(f"Total Hot Spots Per Event  : {avg_hs:.1f}")
        print(f"Mean Execution Time        : {avg_time:.2f} ms")
        print(f"Execution Standard Dev     : {std_time:.2f} ms")
        print(f"Estimated Rate             : {1000.0 / avg_time:.1f} events/sec")
        print("="*45 + "\n")