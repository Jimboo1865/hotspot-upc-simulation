import numpy as np
from config import SimulationConfig

class TargetGeometry:
    def __init__(self, config: SimulationConfig, rng=None):
        self.config = config
        self.rng = rng if rng is not None else np.random.default_rng()

    def _sample_zero_truncated_poisson(self, mean, size=None):
        out = self.rng.poisson(mean, size=size)
        if size is None:
            while out == 0:
                out = self.rng.poisson(mean)
            return int(out)
            
        out = np.asarray(out)
        mask = (out == 0)
        while np.any(mask):
            out[mask] = self.rng.poisson(mean, size=np.sum(mask))
            mask = (out == 0)
        return out
        
    def _woods_saxon_density(self, r):
        return 1.0 / (1.0 + np.exp((r - self.config.R_WS) / self.config.a_WS))
        
    def _sample_woods_saxon_nucleons(self):
        positions = []
        while len(positions) < self.config.A:
            batch_size = max(1000, 2 * (self.config.A - len(positions)))
            u = self.rng.random(batch_size)
            r = self.config.r_max * (u**(1/3))

            accept_prob = self._woods_saxon_density(r)
            accepted_r = r[self.rng.random(batch_size) < accept_prob]

            for rr in accepted_r:
                cos_theta = self.rng.uniform(-1,1)
                sin_theta = np.sqrt(1.0 - cos_theta**2)
                phi = self.rng.uniform(0, 2 * np.pi)

                x = rr * sin_theta * np.cos(phi)
                y = rr * sin_theta * np.sin(phi)
                positions.append([x, y])
                if len(positions) == self.config.A:
                    break
        return np.array(positions)