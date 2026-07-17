import time
import numpy as np
from config import SimulationConfig, MesonParameters, PARTICLE_MAP
from profiles import HotSpotProfileGenerator
from data_logger import PreformanceLogger

def main1():
    print("Initializing STARlight Fluctuating Hot-Spot Framework...")

    j_psi_params = MesonParameters(**PARTICLE_MAP[443])
    config = SimulationConfig(
        target_type="Lead",
        W=100.0,
        meson=j_psi_params
    )

    generator = HotSpotProfileGenerator(config, grid_size=100, grid_max_fm=15.0)
    logger = PreformanceLogger()

    test_runs = 5
    print(f"Starting pipeline benchmark run ({test_runs} iterations)...")

    for i in range(test_runs):
        start = time.perf_counter()
        event_data = generator.generate_event(epsilon=1e-5)
        end = time.perf_counter()

        logger.log_events(start, end, event_data["num_hotspots"])
        print(f" -> Event {i+1} successfully evaluated. Hotspots generated: {event_data['num_hotspots']}")

        assert not np.isnan(event_data["y_C"]).any(), "Calculation contains unexpected NaN values!"

    logger.print_summary()

def run_benchmark(mode_name, grid_size, test_runs, config):
    print(f"Running Benchmark: {mode_name} (Grid: {grid_size if grid_size else 'N/A'})...")

    mode_type = "continuous" if grid_size is None else "discrete"
    generator = HotSpotProfileGenerator(config, grid_size=grid_size or 100, mode=mode_type)
    logger = PreformanceLogger()

    for _ in range(test_runs):
        start = time.perf_counter()
        event_data = generator.generate_event(epsilon=1e-5)
        end = time.perf_counter()

        logger.log_events(start, end, event_data["num_hotspots"])

    logger.print_summary()

def main2():
    print("Initializing STARlight Fluctuating Hot-Spot Framework Benchmark...")

    j_psi_params = MesonParameters(**PARTICLE_MAP[443])
    config = SimulationConfig(
        target_type="Lead",
        W=100.0,
        meson=j_psi_params
    )

    test_runs = 20  # Increased for a more reliable statistical sample size
    
    # Define experimental test-bench configurations
    benchmarks = [
        {"name": "Continuous Mode", "grid": None},
        {"name": "Discrete Low-Res", "grid": 50},
        {"name": "Discrete Mid-Res", "grid": 100},
        {"name": "Discrete High-Res", "grid": 200},
    ]

    for bench in benchmarks:
        run_benchmark(bench["name"], bench["grid"], test_runs, config)


if __name__ == "__main__":
    main2()