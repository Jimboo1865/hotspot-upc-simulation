from dataclasses import dataclass
import numpy as np

HBARC = 0.1973269804  # GeV fm

#Temp Settings Eventuall parse form Slight.in

@dataclass
class MesonParameters:
    M: float
    m_f: float
    e_f: float
    N_T: float
    N_L: float
    Rsquared: float
    a_2S: float = None
@dataclass
class SimulationConfig:
    target_type: str
    W: float
    meson: MesonParameters

    particle_id: int = 443 # J/psi

    #Constants
    Q2: float = 0.05 # Photon Virtuality
    Q0_squared: float = 1 # Arbitrary as set in paper
    x0: float = 2e-4 # Paper source 19 conclusion
    Lambda: float = 0.21 # Paper source 19 conclusion

    # Transverse Area
    B_p_GeV2: float = 4.75 #Proton in GeV^-2
    B_hs_GeV2: float = 0.8 #Hotspot in GeV^-2

    #Fitting Parameters
    p0: float = 0.015
    p1: float = -0.58
    p2: float = 300

    #Lead Parameters
    A: int = 208
    r_max: float = 15.0 # fm

    # Wood-Saxon Parameters
    R_WS: float = 6.62 #fm
    a_WS: float = 0.546 #fm

    @property # Fine-structure constant
    def alpha_em(self) -> float:
        return 1.0 / 137.03599911
    @property # e = sqrt( 4 * pi * alpha_em)
    def elementary_charge(self) -> float:
        return np.sqrt(4.0 * np.pi * self.alpha_em)
    @property #B_p in fm
    def sigma_hotspot_position_fm(self):
        return np.sqrt(self.B_p_GeV2 * HBARC**2)
    @property #B_hs in fm
    def sigma_hotspot_shape_fm(self) -> float:
        return np.sqrt(self.B_hs_GeV2 * HBARC**2)
    @property 
    def bjorken_x(self) -> float:
        return (self.Q2 + self.meson.M**2) / (self.Q2 + self.W**2)
    @property
    def saturation_scale(self) -> float:
        return (self.Q0_squared * (self.x0 / self.bjorken_x)**self.Lambda)
    @property
    def sigma_0(self) -> float:
        return (4 * np.pi * self.sigma_hotspot_position_fm**2)
    @property
    def mean_hotspots(self) -> float:
        return self.p0 * self.bjorken_x**self.p1 * (1 + (self.p2 * np.sqrt(self.bjorken_x)))
    
PARTICLE_MAP = {
    113: { # rho0
        "M": 0.775260, "m_f": 0.14, "e_f": 1 / np.sqrt(2),
        "N_T": 0.909, "N_L": 0.853, "Rsquared": 12.75, "a_2S": None
        },
    333: { # phi
        "M": 1.019461, "m_f": 0.14, "e_f": 1 / 3,
        "N_T": 0.918, "N_L": 0.823, "Rsquared": 11.3, "a_2S": None
        },
    443: { # J/psi
        "M": 3.09690, "m_f": 1.4, "e_f": 2 / 3,
        "N_T": 0.582, "N_L": 0.578, "Rsquared": 2.24, "a_2S": None
        },
    100443: { # psi(2S)
        "M": 3.686097, "m_f": 1.4, "e_f": 2 / 3,
        "N_T": 0.666, "N_L": 0.658, "Rsquared": 3.705, "a_2S": -0.6225
        },
    553: { # Upsilon(1S)
        "M": 9.46030, "m_f": 4.2, "e_f": 1 / 3,
        "N_T": 0.478, "N_L": 0.478, "Rsquared": 0.585, "a_2S": None
        },
    100553: { # Upsilon(2S)
        "M": 10.02326, "m_f": 4.2, "e_f": 1 / 3,
        "N_T": 0.614, "N_L": 0.610, "Rsquared": 0.831, "a_2S": -0.568
        }
}