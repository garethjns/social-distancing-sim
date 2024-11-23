import copy
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, Hashable, List, Union

import numpy as np


@dataclass
class Disease:
    """

    :param immunity_mean: Mean prop. immunity gained after survival.
    :param immunity_std:
    :param immunity_decay_mean: Mean prop. immunity decay per step.
    """

    seed: Union[None, int] = None
    name: str = "COVID-19"
    virulence: float = 0.006
    recovery_rate: float = 0.98
    duration_mean: float = 21
    duration_std: float = 5
    immunity_mean: float = 0.8
    immunity_std: float = 0.02
    immunity_decay_mean: float = 0.1
    immunity_decay_std: float = 0.005

    _infected_key: str = field(init=False, default="infected")
    _mask_key: str = field(init=False, default="mask")
    _immune_key: str = field(init=False, default="immune")
    _alive_key: str = field(init=False, default="alive")

    def __post_init__(self) -> None:
        self._prepare_random_state()

    def _prepare_random_state(self) -> None:
        self.state = np.random.RandomState(seed=self.seed)

    def give_immunity(
        self, node: Dict[Hashable, Any], immunity: float = None
    ) -> Dict[Hashable, Any]:
        """
        Give immunity to a node.

        :param node: Graph node data.
        :param immunity: Amount of immunity to give to node, optional. Default None, which gives self.immunity_ (the
                         values applied on recovery). Allows for bonus immunity from actions, like vaccination.
        :return: Modified node data.
        """
        if immunity is None:
            immunity = min(
                self.immunity_mean + self.state.normal(scale=self.immunity_std), 1.0
            )
        node[self._immune_key] = immunity

        return node

    def decay_immunity(self, node: Dict[Hashable, Any]) -> Dict[Hashable, Any]:
        """
        Decay immunity on a node.

        Uses self.immunity_decay_ values. Not option to specify decay amount for now, because YAGNI.

        :param node: Graph node data.
        :return: Modified node data.
        """
        decay = self.immunity_decay_mean + self.state.normal(
            scale=self.immunity_decay_std
        )
        new_immunity = max(0.0, node[self._immune_key] - node[self._immune_key] * decay)
        node[self._immune_key] = new_immunity

        return node

    @staticmethod
    def _modify_virulence(original: float, modifier: float) -> float:
        return min(max(1e-7, original * (1 - modifier)), 0.999)

    def modified_virulence(self, modifiers: Union[float, List[float]]) -> float:
        """Reduce baseline disease virulence according to modifier(s), i.e. immunity and/or masks."""
        if not isinstance(modifiers, list):
            modifiers = [modifiers]

        vir = self.virulence
        for mod in modifiers:
            vir = self._modify_virulence(vir, mod)

        return vir

    def conclude(
        self,
        node: Dict[Hashable, Any],
        chance_to_force: float = 0.0,
        recovery_rate_modifier: float = 1,
    ) -> Dict[Hashable, Any]:
        """

        :param node: Graph node to update.

        :param chance_to_force: Chance to force conclusion. Must be between 0 -> 1. Optional, default 0.
                                Default uses self.duration_* params instead. Allows for specifying a treatment efficacy
                                along with recovery rate modified. Note that if conclusion is forced, node can still
                                die. So treatment can be worse than cure if high chance to force conclusion and
                                recovery rate penalty.
        :param recovery_rate_modifier: Modify recovery rate depending on external factors such as healthcare burden.
                                       Default 1.0 (no modification). Only relevant if disease ends.

        :return: Updated graph node.
        """

        # Decide if forcing conclusion or not
        force = False
        if chance_to_force > 0:
            force = self.state.binomial(1, chance_to_force)

        # Decide end of disease
        if (
            node[self._infected_key]
            > self.state.normal(self.duration_mean, self.duration_std, size=1)
            or force
        ):
            # Concluding, decide fate
            node[self._infected_key] = 0
            modified_recovery_rate = max(
                0.0, min(1.0, self.recovery_rate * recovery_rate_modifier)
            )

            if self.state.binomial(1, modified_recovery_rate):
                node = self.give_immunity(node)
                node[self._alive_key] = True
            else:
                node[self._alive_key] = False

        else:
            # Continue disease progression
            node[self._infected_key] += 1

        return node

    def try_to_infect_multiple(
        self, source_node: Dict[Hashable, Any], target_nodes: List[Dict[Hashable, Any]]
    ) -> List[int]:

        # Source node not infected so can't infect anyone
        if source_node.get(self._infected_key, 0) < 1:
            warnings.warn(
                "Attempted to infect with non-infectious source, returning no new infections."
            )
            return [0] * len(target_nodes)

        # Prepare virulence wrt target node. 0 if target node is already infected
        infect_status = [t_node.get(self._infected_key, 0) for t_node in target_nodes]
        virulence_status = [
            (
                self.modified_virulence(
                    modifiers=[
                        t_node.get(self._immune_key, 0),
                        t_node.get(self._mask_key, 0),
                        source_node.get(self._mask_key, 0),
                    ]
                )
                if not already_infected
                else 0
            )
            for t_node, already_infected in zip(target_nodes, infect_status)
        ]

        # Sample and apply new statuses
        new_infection_status = self.state.binomial(1, virulence_status)
        for tar_node, status in zip(target_nodes, new_infection_status):
            if status > 0:
                tar_node[self._infected_key] = int(status)

        return list(new_infection_status)

    def try_to_infect(
        self, source_node: Dict[Hashable, Any], target_node: Dict[Hashable, Any]
    ) -> Dict[Hashable, Any]:
        """
        Attempt to infect another node.

        Depends on 3 factors:
         - The immunity of the target node - uses target nodes immunity
         - If the infected node is masked - uses own mask modifier
         - If the uninfected node is masked - uses target nodes mask modifier
         TODO: Considering a bonus to protection if both are masked, but not implemented for now. It's a bit
               speculative.
        """
        if (target_node.get(self._infected_key, 0) <= 0) and (
            source_node.get(self._infected_key, 0) > 0
        ):
            vir = self.modified_virulence(
                modifiers=[
                    target_node.get(self._immune_key, 0),
                    target_node.get(self._mask_key, 0),
                    source_node.get(self._mask_key, 0),
                ]
            )

            target_node[self._infected_key] = self.state.binomial(1, vir)

        return target_node

    def force_infect(self, node: Dict[Hashable, Any]) -> Dict[Hashable, Any]:
        node[self._infected_key] = 1

        return node

    def clone(self):
        clone = copy.deepcopy(self)
        clone._prepare_random_state()

        return clone
