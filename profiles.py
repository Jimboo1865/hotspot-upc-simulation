import numpy as np
from config import SimulationConfig
from geometry import TargetGeometry

class HotSpotProfileGenerator:
    def __init__(self, config: SimulationConfig, grid_size: int = 100, grid_max_fm: float = 15.0, mode: str = "discrete"):
        self.config = config
        self.geom = TargetGeometry(config)
        self.grid_size = grid_size
        self.grid_max_fm = grid_max_fm
        self.mode = mode.lower()

        if self.mode == "discrete":
            self.xy = np.linspace(-grid_max_fm, grid_max_fm, grid_size)
            # Calculate the uniform grid spacing step
            self.dx = self.xy[1] - self.xy[0] 
            
            self.X, self.Y = np.meshgrid(self.xy, self.xy)
            # Shapes: (grid_size, grid_size, 1, 2) for clean broadcasting
            self.grid_points = np.stack([self.X, self.Y], axis=-1)[:, :, np.newaxis, :]
            
            # Pre-compute fixed Wood-Saxon baseline profile to avoid recreating it every single event
            self.r_grid = np.sqrt(self.X**2 + self.Y**2)
            self.T_WS = 1.0 / (1.0 + np.exp((self.r_grid - self.config.R_WS) / self.config.a_WS))
            # 2D Integration using flat sum multiplied by the area element dx^2
            self.integral_T_WS = np.sum(self.T_WS) * (self.dx ** 2)
            self.T_WS_hat = self.T_WS / (self.integral_T_WS + 1e-12)

    def generate_event(self, epsilon: float = 1e-5) -> dict:
        # 1. Sample Nucleon Centers
        nucleon_centers = self.geom._sample_woods_saxon_nucleons()

        # 2. Sample Hotspots per Nucleon
        mean_hs = self.config.mean_hotspots
        hs_counts = self.geom._sample_zero_truncated_poisson(mean_hs, size=self.config.A)
        total_hotspots = int(np.sum(hs_counts))

        # 3. Generate Hotspot Positions
        repeated_centers = np.repeat(nucleon_centers, hs_counts, axis=0)
        sigma_p = self.config.sigma_hotspot_position_fm
        hs_offsets = self.geom.rng.normal(0, sigma_p, size=(total_hotspots, 2))
        hs_positions = repeated_centers + hs_offsets

        # --- CONTINUOUS MODE ---
        if self.mode == "continuous":
            return {
                "hs_positions": hs_positions,
                "num_hotspots": total_hotspots,
                "y_C": np.array([0.0]) 
            }

        # --- DISCRETE MESHGRID MODE ---
        # Broadcasting: (G, G, 1, 2) - (1, 1, H, 2) -> (G, G, H, 2)
        delta_sq = np.sum((self.grid_points - hs_positions[np.newaxis, np.newaxis, :, :])**2, axis=-1)

        sigma_hs = self.config.sigma_hotspot_position_fm
        gaussians = np.exp(-delta_sq / (2 * sigma_hs**2)) / (2 * np.pi * sigma_hs**2)
        T_g_C = np.sum(gaussians, axis=-1)

        # Compute 2D profile integral efficiently
        integral_Tg = np.sum(T_g_C) * (self.dx ** 2)
        T_g_C_hat = T_g_C / (integral_Tg + 1e-12)

        # Using pre-cached T_WS_hat calculations
        R_g_C = T_g_C_hat / (self.T_WS_hat + epsilon)
        y_C = np.log(R_g_C + epsilon)

        return {
            "X": self.X,
            "Y": self.Y,
            "T_g_C_hat": T_g_C_hat,
            "T_WS_hat": self.T_WS_hat,
            "y_C": y_C,
            "num_hotspots": total_hotspots
        }