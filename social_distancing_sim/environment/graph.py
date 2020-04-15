import copy
from dataclasses import dataclass
from typing import Callable
from typing import Dict
from typing import List, Union

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns

from social_distancing_sim.environment.history import History


@dataclass
class Graph:
    """Class to handle environment graph, generation, etc."""
    seed: Union[int, None] = None
    layout: str = "spring_layout"

    community_n: int = 5
    community_size_mean: int = 5
    community_size_std: int = 1
    community_p_in: float = 0.2
    community_p_out: float = 0.1

    considered_immune_threshold: float = 0.3

    def __post_init__(self):
        self._prepare_random_state()

        self._layout: Callable = getattr(nx, self.layout)

        self._community_sizes: np.ndarray = self._random_state.poisson(self._random_state.normal(size=self.community_n)
                                                                       * self.community_size_std
                                                                       + self.community_size_mean)
        self.g_: nx.classes.graph.Graph
        self.g_pos_: Union[None, Dict[int, np.ndarray]] = None
        self._generate_graph()
        self.reset_cached_values()

    def reset_cached_values(self):
        self._current_infected_nodes: Union[int, None] = None
        self._current_isolated_nodes: Union[int, None] = None
        self._current_immune_nodes: Union[int, None] = None
        self._current_clear_nodes: Union[int, None] = None
        self._current_alive_nodes: Union[int, None] = None
        self._current_dead_nodes: Union[int, None] = None

    @property
    def total_population(self) -> int:
        return len(self.g_.nodes)

    def _generate_graph(self) -> None:
        """Creates the networkx random partition graph."""
        self.g_ = nx.random_partition_graph(list(self._community_sizes),
                                            p_in=self.community_p_in,
                                            p_out=self.community_p_out,
                                            seed=self.seed)

        for _, nv in self.g_.nodes.data():
            nv["infected"] = 0
            nv["immune"] = False
            nv["alive"] = True

    def _prepare_random_state(self) -> None:
        self._random_state = np.random.RandomState(seed=self.seed)

    @property
    def n_current_infected(self) -> int:
        return len(self.current_infected_nodes)

    @property
    def current_isolated_nodes(self) -> List[int]:
        if self._current_isolated_nodes is None:
            self._current_isolated_nodes = [nk for nk, nv in self.g_.nodes.data() if nv.get("isolated", False)]
        return self._current_isolated_nodes

    @property
    def current_infected_nodes(self) -> List[int]:
        if self._current_infected_nodes is None:
            self._current_infected_nodes = [nk for nk, nv in self.g_.nodes.data() if (nv["infected"] > 0) & nv["alive"]]
        return self._current_infected_nodes

    @property
    def current_immune_nodes(self) -> List[int]:
        if self._current_immune_nodes is None:
            self._current_immune_nodes = [nk for nk, nv in self.g_.nodes.data()
                                          if (nv["immune"] >= self.considered_immune_threshold) & nv["alive"]]
        return self._current_immune_nodes

    @property
    def current_clear_nodes(self) -> List[int]:
        if self._current_clear_nodes is None:
            self._current_clear_nodes = [nk for nk, nv in self.g_.nodes.data() if (nv["infected"] == 0) & nv["alive"]]
        return self._current_clear_nodes

    @property
    def current_alive_nodes(self) -> List[int]:
        if self._current_alive_nodes is None:
            self._current_alive_nodes = [nk for nk, nv in self.g_.nodes.data() if nv["alive"]]
        return self._current_alive_nodes

    @property
    def current_dead_nodes(self) -> List[int]:
        if self._current_dead_nodes is None:
            self._current_dead_nodes = [nk for nk, nv in self.g_.nodes.data() if not nv["alive"]]
        return self._current_dead_nodes

    @property
    def overall_death_rate(self) -> float:
        if len(self.current_dead_nodes) > 0:
            death_rate = len(self.current_dead_nodes) / self.total_population
        else:
            death_rate = 0

        return death_rate

    def isolate_node(self, node_id: int,
                     effectiveness: float = 0.95):
        """
        Remove some or all edges from a node, and store on node.

        :param node_id: Node index.
        :param effectiveness: Proportion of edges to remove
        """
        node = self.g_.nodes[node_id]
        # Do NOT deepcopy EdgeView!!! Copy won't work either.
        node["_edges"] = copy.deepcopy(list(self.g_.edges(node_id)))
        node["isolated"] = True

        # Select edges to remove
        to_remove = []
        for uv in self.g_.edges(node_id):
            if self._random_state.binomial(1, effectiveness):
                to_remove.append(uv)

        self.g_.remove_edges_from(to_remove)

    def reconnect_node(self, node_id: int):
        node = self.g_.nodes[node_id]
        self.g_.add_edges_from(node["_edges"])
        node["isolated"] = False

    def plot(self,
             ax: Union[None, plt.Axes] = None,
             history: History = None) -> None:
        """
        Plot the full network graph.

        :param ax: Matplotlib to draw on.
        :param history: If provided, gets colours from defaults set in history, if available.
        """
        sns.set()

        if history is not None:
            colours = history.colours
        else:
            colours = {}

        if self.g_pos_ is None:
            self.g_pos_ = self._layout(self.g_,
                                       seed=self.seed)

        nx.draw_networkx_nodes(self.g_, self.g_pos_,
                               nodelist=self.current_clear_nodes,
                               node_color=colours.get('Current clear', '#1f77b4'),
                               node_size=10,
                               ax=ax)
        nx.draw_networkx_nodes(self.g_, self.g_pos_,
                               nodelist=self.current_immune_nodes,
                               node_color=colours.get('Total immune', '#9467bd'),
                               node_size=10,
                               ax=ax)
        nx.draw_networkx_nodes(self.g_, self.g_pos_,
                               nodelist=self.current_infected_nodes,
                               node_color=colours.get('Current infections', '#d62728'),
                               node_size=10,
                               ax=ax)
        nx.draw_networkx_nodes(self.g_, self.g_pos_,
                               nodelist=self.current_dead_nodes,
                               node_color=colours.get('Total deaths', 'k'),
                               node_size=10,
                               ax=ax)
        nx.draw_networkx_edges(self.g_, self.g_pos_,
                               width=1 / (self.total_population / 5),
                               ax=ax)

    def clone(self) -> "Graph":
        """Clone a fresh object with same seed (could be None)."""
        return Graph(seed=self.seed,
                     layout=self.layout,
                     community_n=self.community_n,
                     community_size_mean=self.community_size_mean,
                     community_size_std=self.community_size_std,
                     community_p_in=self.community_p_in,
                     community_p_out=self.community_p_out,
                     considered_immune_threshold=self.considered_immune_threshold)
