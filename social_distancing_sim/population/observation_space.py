from dataclasses import dataclass
from typing import List, Union

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns

from social_distancing_sim.population.graph import Graph
from social_distancing_sim.population.history import History


@dataclass
class ObservationSpace:
    """
    Wraps a graph to limit observable space through testing.

    Access methods and properties via ObservationSpace to limit access, use .graph to access full graph. Caller has
    has responsibility!

    If test rate is >= 1 observation space has full access to graph.
    """
    graph: Graph
    test_rate: float = 1
    test_validity_period: float = 5
    seed: Union[None, int] = None

    def __post_init__(self) -> None:
        self._prepare_random_state()
        self.reset_cached_values()

    def reset_cached_values(self):
        self._unknown_nodes: Union[int, None] = None
        self._known_current_infected_nodes: Union[int, None] = None
        self._known_current_immune_nodes: Union[int, None] = None
        self._known_current_clear_nodes: Union[int, None] = None

    def _prepare_random_state(self) -> None:
        self.state = np.random.RandomState(seed=self.seed)

    @property
    def known_n_current_infected(self):
        return len(self.known_current_infected_nodes)

    @property
    def unknown_nodes(self) -> List[int]:
        if self._unknown_nodes is None:
            self._unknown_nodes = [nk for nk, nv in self.graph.g_.nodes.data() if (nv.get("status", '') == '')]
        return self._unknown_nodes

    @property
    def known_current_infected_nodes(self) -> List[int]:
        if self._known_current_infected_nodes is None:
            if self.test_rate >= 1:
                self._known_current_infected_nodes = self.graph.current_infected_nodes
            else:
                self._known_current_infected_nodes = [nk for nk, nv in self.graph.g_.nodes.data()
                                                      if (nv.get("status", '') == 'infected')]
        return self._known_current_infected_nodes

    @property
    def known_current_immune_nodes(self) -> List[int]:
        if self._known_current_immune_nodes is None:
            if self.test_rate >= 1:
                self._known_current_immune_nodes = self.graph.current_immune_nodes
            else:
                self._known_current_immune_nodes = [nk for nk, nv in self.graph.g_.nodes.data()
                                                    if (nv.get("status", '') == 'immune')]
        return self._known_current_immune_nodes

    @property
    def known_current_clear_nodes(self) -> List[int]:
        if self._known_current_clear_nodes is None:
            if self.test_rate >= 1:
                self._known_current_clear_nodes = self.graph.current_clear_nodes
            else:
                self._known_current_clear_nodes = [nk for nk, nv in self.graph.g_.nodes.data()
                                                   if (nv.get("status", '') == 'clear')]
        return self._known_current_clear_nodes

    def test_population(self, time_step: int) -> None:
        """
        Test random members of the population, based on testing rate.

        Chance of testing infected is grater than testing asymptomatic.
        """
        clear_test_rate = self.test_rate / 2
        infected_test_rate = self.test_rate * 2

        if clear_test_rate > 1:
            clear_test_rate = 1
        if infected_test_rate > 1:
            infected_test_rate = 1

        for n in self.graph.current_clear_nodes:
            if self.state.binomial(1, clear_test_rate):
                self.graph.g_.nodes[n]['last_tested'] = time_step

        for n in self.graph.current_infected_nodes:
            if self.state.binomial(1, infected_test_rate):
                self.graph.g_.nodes[n]['last_tested'] = time_step

    def update_observed_statuses(self, time_step: int) -> int:
        known_new_infections = 0

        for nk, nv in self.graph.g_.nodes.data():
            # Is dead
            if not nv['alive']:
                nv['status'] = 'dead'
                continue

            # Is known immune, stays immune, stays alive, never updated
            if nv.get("status", "") == "immune":
                continue

            # Only propagate immune status if we knew node was infected
            if nv['immune'] and (nv.get("status", "") == "infected"):
                nv['status'] = 'immune'
                continue

            # Is infected and tested this turn
            if nv['infected'] > 0 and (nv.get("last_tested", -1) == time_step):
                nv['status'] = 'infected'
                known_new_infections += 1

            # Is clear and tested this turn
            if nv['infected'] == 0 and (nv.get("last_tested", -1) == time_step):
                nv['status'] = 'clear'

            # Test has expired (only for clear nodes)
            if ((nv.get("status", '') == "clear")
                    and ((time_step - nv.get("last_tested", 0)) > self.test_validity_period)):
                nv['status'] = ''

        return known_new_infections

    def plot(self,
             ax: Union[None, plt.Axes] = None,
             history: History = None) -> None:
        """
        Plot the observed network graph.

        :param ax: Matplotlib to draw on.
        :param history: If provided, gets colours from defaults set in history, if available.
        """
        sns.set()

        if history is not None:
            colours = history.colours
        else:
            colours = {}

        if self.graph.g_pos_ is None:
            self.graph.g_pos_ = nx.spring_layout(self.graph.g_,
                                                 seed=self.seed)

        nx.draw_networkx_nodes(self.graph.g_, self.graph.g_pos_,
                               nodelist=self.unknown_nodes,
                               node_color=colours.get('Unknown', '#bdbcbb'),
                               node_size=10,
                               ax=ax)
        nx.draw_networkx_nodes(self.graph.g_, self.graph.g_pos_,
                               nodelist=self.known_current_clear_nodes,
                               node_color=colours.get('Known current clear', '#1f77b4'),
                               node_size=10,
                               ax=ax)
        nx.draw_networkx_nodes(self.graph.g_, self.graph.g_pos_,
                               nodelist=self.known_current_immune_nodes,
                               node_color=colours.get('Known total immune', '#9467bd'),
                               node_size=10,
                               ax=ax)
        nx.draw_networkx_nodes(self.graph.g_, self.graph.g_pos_,
                               nodelist=self.known_current_infected_nodes,
                               node_color=colours.get('Known current infections', '#d62728'),
                               node_size=10,
                               ax=ax)
        nx.draw_networkx_nodes(self.graph.g_, self.graph.g_pos_,
                               nodelist=self.graph.current_dead_nodes,
                               node_color=colours.get('Total deaths', 'k'),
                               node_size=10,
                               ax=ax)
        nx.draw_networkx_edges(self.graph.g_, self.graph.g_pos_,
                               width=0.01,
                               ax=ax)
