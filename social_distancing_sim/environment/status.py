from typing import Union, Tuple, List

import numpy as np


class Status:
    """
    Handles logic of visible node status.

    1) If not alive, dead
    2) If infected, not clear or immune
    3) If clear, not infected, but not necessarily immune
    4) If immune, not infected
    5) If recovered, clear, immune, not infected
    6) If isolated

    """

    def __init__(self,
                 alive: Union[bool, None] = True,
                 infected: Union[bool, None] = None,
                 clear: Union[bool, None] = None,
                 isolated: Union[bool, None] = None,
                 immune: Union[bool, None] = None,
                 last_tested: Union[int, None] = -999):

        if infected is not None:
            if clear is not None:
                if infected & clear:
                    raise ValueError("Node cannot have status infected and clear")
            if immune is not None:
                if infected & immune:
                    raise ValueError("Node cannot have status infected and immune")

        self._clear = None
        self._infected = None
        self._immune = None
        self._alive = None

        # Don't set Nones to avoid overwriting behaviour, eg if clear=False but infected=None
        # (Setters still set Nones)
        self.alive = alive
        if clear is not None:
            self.clear = clear
        if infected is not None:
            self.infected = infected
        self._isolated = isolated
        if immune is not None:
            self.immune = immune

        self.last_tested: int = last_tested

    def __repr__(self) -> str:
        return f"Status(alive={self.alive}, infected={self.infected}, clear={self.clear}, isolated={self.isolated}, " \
               f"immune={self.immune}, last_tested={self.last_tested})"

    def __hash__(self) -> int:
        return hash(self.__repr__())

    def __eq__(self, other: "Status") -> bool:
        return self.__hash__() == other.__hash__()

    def set_health_unknown(self):
        self._clear = None
        self._infected = None
        self._immune = None

    @property
    def alive(self) -> bool:
        return self._alive

    @alive.setter
    def alive(self, flag: bool):
        self._alive = flag
        if not self._alive:
            self._recovered = False
            self._clear = False
            self._infected = False
            self._isolated = False
            self._immune = False

    @property
    def dead(self) -> bool:
        """Opposite of alive."""
        return not self.alive

    @property
    def infected(self) -> bool:
        return self._infected

    @infected.setter
    def infected(self, flag: bool):
        """Infected -> True voids immunity and clear"""
        if flag is not None:
            self.clear = not flag

            if flag:
                self.immune = False

        self._infected = flag

    @property
    def clear(self) -> bool:
        return self._clear

    @clear.setter
    def clear(self, flag: bool):
        # Can't be clear and infected
        if flag is not None:
            self._infected = not flag

            if not flag:
                self.immune = False

        self._clear = flag

    @property
    def recovered(self):
        return self._recovered

    @recovered.setter
    def recovered(self, flag: bool):
        """Sets to clear and immune"""
        self.clear = flag

    @property
    def isolated(self) -> bool:
        """Can be isolated and anything else."""
        return self._isolated

    @isolated.setter
    def isolated(self, flag: bool):
        self._isolated = flag

    @property
    def immune(self) -> bool:
        """Immunity can be assumed, but not amount"""
        return self._immune

    @immune.setter
    def immune(self, flag: bool):
        if flag is not None:
            if flag:
                self.clear = True

        self._immune = flag

    @property
    def state_features_names(self) -> List[str]:
        """Status fields to use to co construct state."""
        return ['alive', 'clear', 'infected', 'immune', 'isolated']

    @property
    def state(self) -> np.ndarray:
        """Convert node status into state array."""
        return np.array([getattr(self, a) for a in self.state_features_names],
                        dtype=bool)


if __name__ == "__main__":
    status = Status()
