import copy
from dataclasses import dataclass

from social_distancing_sim.population.graph import Graph
from social_distancing_sim.population.observation_space import ObservationSpace


@dataclass
class Scoring:
    """
    Class holding values/methods used for scoring population (not action costs).

    Each alive, clear node yields a number of points depending on their number of connections. Doesn't matter what
    they're connected to. This means actions like isolation (even if free), and events like infection and death have
    ongoing costs.

    Death has an additional cost defined in death penalty
    """
    clear_yield_per_edge: int = 0.1
    infection_penalty: int = -1
    death_penalty: int = -200

    def score(self, graph: [Graph, ObservationSpace],
              new_infections: int = 0,
              new_deaths: int = 0) -> float:

        infection_penalty = new_infections * self.infection_penalty
        death_penalty = new_deaths * self.death_penalty

        if isinstance(graph, Graph):
            g_ = graph.g_
        else:
            g_ = graph.graph.g_

        clear_yield = 0
        for node_id in graph.current_clear_nodes:
            clear_yield += self.clear_yield_per_edge * len(list(g_.neighbors(node_id)))

        score = infection_penalty + clear_yield + death_penalty

        return score

    def clone(self) -> "Scoring":
        """Clone, nothing stochastic to handle."""
        return copy.deepcopy(self)
