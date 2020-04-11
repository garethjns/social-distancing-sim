import copy
from dataclasses import dataclass
from functools import lru_cache


@dataclass
class Healthcare:
    capacity: int = 200
    max_penalty: float = 0.5

    def __hash__(self):
        return hash(self.__repr__())

    @lru_cache(maxsize=1024)
    def recovery_rate_penalty(self, n_current_infected: int) -> float:
        """
        Penalise the disease survivability by current healthcare burden.

        If utilisation is below capacity, no penalty. If above, reduce survivability by some function of this.
        Limited to minimum of 50% normal recovery rate.
        """
        # Set current healthcare penalty
        if n_current_infected <= self.capacity:
            # No penalty
            recovery_rate_penalty = 1
        else:
            recovery_rate_penalty = min(1.0, self.capacity / n_current_infected)

        # Reduce disease survivability by this %
        return max(recovery_rate_penalty, self.max_penalty)

    def clone(self):
        return copy.deepcopy(self)
