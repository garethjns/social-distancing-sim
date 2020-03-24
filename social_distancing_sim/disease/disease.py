from dataclasses import dataclass


@dataclass
class Disease:
    virulence: float = 0.005
    recovery_rate: float = 0.98
    duration_mean: float = 21
    duration_std: float = 5
    name: str = 'COVID-19'
