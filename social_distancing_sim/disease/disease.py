from dataclasses import dataclass
from typing import Union

import numpy as np


@dataclass
class Disease:
    seed: Union[None, int] = None
    virulence: float = 0.006
    recovery_rate: float = 0.98
    duration_mean: float = 21
    duration_std: float = 5
    name: str = 'COVID-19'

    def __post_init__(self):
        self._prepare_random_state()

    def _prepare_random_state(self) -> None:
        self.state = np.random.RandomState(seed=self.seed)

    def modified_virulence(self, immunity: float) -> float:
        """Reduce virulence according to immunity"""
        return min(max(1e-7, self.virulence * (1 - immunity)), 0.999)

    def conclude(self, node,
                 recovery_rate_modifier: float = 1):
        """

        :param node: Graph node to update.
        :param recovery_rate_modifier: Modify recovery rate depending on external factors such as healthcare burden.
                                       Default 1.0 (no modification).
        :return: Updated graph node.
        """

        modified_recovery_rate = self.recovery_rate * recovery_rate_modifier

        # Decide end of disease
        if node["infected"] > self.state.normal(self.duration_mean,
                                                self.duration_std,
                                                size=1):
            node["infected"] = 0
            node["immune"] = True

            if self.state.binomial(1, modified_recovery_rate):
                node["alive"] = True
            else:
                node["alive"] = False

        else:
            # Continue disease progression
            node["infected"] += 1

        return node

    def try_to_infect(self, node):
        if not node.get("infected", 0) > 0:
            node["infected"] = self.state.binomial(1, self.modified_virulence(node.get("immune", 0)))
        return node

    @staticmethod
    def force_infect(node):
        node['infected'] = 1
        return node
