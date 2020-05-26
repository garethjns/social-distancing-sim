from dataclasses import dataclass
from typing import Callable
from typing import Dict
from typing import List, Union

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns


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
        self._current_masked_nodes: Union[int, None] = None

    def state_summary(self) -> np.ndarray:
        """Vector representing n of each node type."""
        return np.array([len(self.current_clear_nodes), self.n_current_infected, len(self.current_isolated_nodes),
                         len(self.current_immune_nodes), len(self.current_alive_nodes)])

    def state_graph(self) -> np.ndarray:
        """Node x node matrix representing graph."""
        return nx.convert_matrix.to_numpy_array(self.g_,
                                                dtype=np.int16)

    def state_nodes(self) -> np.ndarray:
        """Node x node_state matrix."""
        return np.array([[nd[c] for c in ["alive", "infected", "immune", "isolated", "masked"]]
                         for nv, nd in self.g_.nodes.data()])

    def state_full(self) -> np.ndarray:
        """.state_nodes + .state_graph"""
        return np.concatenate([self.state_nodes(), self.state_graph()],
                              axis=1)

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
            nv["_edges"] = []
            nv["isolated"] = False
            nv["mask"] = 0.0

    def _prepare_random_state(self) -> None:
        self._random_state = np.random.RandomState(seed=self.seed)

    @property
    def n_current_infected(self) -> int:
        return len(self.current_infected_nodes)

    @property
    def current_masked_nodes(self) -> List[int]:
        if self._current_masked_nodes is None:
            self._current_masked_nodes = [nk for nk, nv in self.g_.nodes.data() if nv.get("mask", 0) > 0]
        return self._current_masked_nodes

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
                     effectiveness: float = 0.95) -> None:
        """
        Remove some or all edges from a node, and store on node.

        Flag node as isolated if any edges removed and stored in _edges.

        :param node_id: Node index.
        :param effectiveness: Proportion of edges to remove
        """
        node = self.g_.nodes[node_id]
        node["isolated"] = True

        # Select edges to remove
        to_remove = []
        for uv in self.g_.edges(node_id):
            if self._random_state.binomial(1, effectiveness):
                to_remove.append(uv)

        # Do NOT deepcopy EdgeView!!! Copy won't work either.
        node["_edges"] += to_remove

        self.g_.remove_edges_from(to_remove)

    def reconnect_node(self, node_id: int,
                       effectiveness: float = 0.95) -> None:
        """
        Restore edges with probability defined in effectiveness.

        Flag node as not isolated when all edges have been restored.

        :param node_id: Node index.
        :param effectiveness: Proportion of edges to re-add.
        """
        node = self.g_.nodes[node_id]
        to_add = []
        leave = []
        for uv in node["_edges"]:
            if self._random_state.binomial(1, effectiveness):
                to_add.append(uv)
            else:
                leave.append(uv)

        self.g_.add_edges_from(to_add)
        node["_edges"] = leave
        if len(node["_edges"]) == 0:
            node["isolated"] = False

    def mask_node(self, node_id: int,
                  effectiveness: float = 0.5) -> None:
        self.g_.nodes[node_id]["mask"] = effectiveness

    def unmask_node(self, node_id: int) -> None:
        self.g_.nodes[node_id]["mask"] = 0

    def plot_matrix(self,
                    ax: Union[None, plt.Axes] = None) -> plt.Figure:
        fig = sns.heatmap(self.state_graph(),
                          ax=ax)

        return fig

    def plot_summary(self,
                     ax: Union[None, plt.Axes] = None) -> plt.Axes:

        ax_ = ax
        for i, c in enumerate(["alive", "infected", "immune", "isolated"]):
            ax_ = sns.distplot(self.state_nodes()[:, i],
                               label=c,
                               kde=False,
                               ax=ax)
        plt.legend()

        return ax_

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


if __name__ == "__main__":
    g = Graph()
    g.state_summary()
    g.state_full()
    g.state_nodes()
    g.state_graph()

    g.plot_summary()
