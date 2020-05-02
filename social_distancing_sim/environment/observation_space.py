from dataclasses import dataclass
from typing import List, Union, Tuple, Dict

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns

from social_distancing_sim.environment.graph import Graph
from social_distancing_sim.environment.status import Status


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
        self._attach_status_to_graph()

    def reset_cached_values(self):
        self._unknown_nodes: Union[int, None] = None
        self._known_nodes: Union[int, None] = None
        self._current_infected_nodes: Union[List[int], None] = None
        self._current_immune_nodes: Union[List[int], None] = None
        self._current_clear_nodes: Union[List[int], None] = None
        self._isolated_nodes: Union[List[int], None] = None

    def _attach_status_to_graph(self):
        for _, nv in self.graph.g_.nodes.data():
            nv["status"] = Status()

    def _prepare_random_state(self) -> None:
        self._random_state = np.random.RandomState(seed=self.seed)

    def state_summary(self) -> np.ndarray:
        """Vector representing n of each node type. Contains known values."""
        return np.array([len(self.current_clear_nodes), len(self.current_infected_nodes),
                         len(self.isolated_nodes), len(self.current_immune_nodes),
                         len(self.current_alive_nodes), len(self.unknown_nodes)])

    def state_graph(self) -> np.ndarray:
        """Node x node matrix representing graph. All connections are known, so same as .graph."""
        return self.graph.state_graph()

    def state_nodes(self) -> np.ndarray:
        """Node x node_state matrix. Uses node["status"] which handles known node state."""
        return np.array([nd["status"].state for _, nd in self.graph.g_.nodes.data()])

    def state_full(self) -> np.ndarray:
        """.state_nodes + .state_graph"""
        return np.concatenate([self.state_nodes(), self.state_graph()],
                              axis=1)

    @property
    def state(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (self.state_summary(),
                self.state_graph(),
                self.state_nodes())

    def plot_matrix(self,
                    ax: Union[None, plt.Axes] = None) -> plt.Figure:
        fig = sns.heatmap(self.state_graph(),
                          ax=ax)

        return fig

    def plot_summary(self,
                     ax: Union[None, plt.Axes] = None) -> plt.Axes:

        ax_ = ax
        for i, c in enumerate(Status().state_features_names):
            ax_ = sns.distplot(self.state_nodes()[:, i],
                               label=c,
                               kde=False,
                               ax=ax)
        plt.legend()

        return ax_

    @property
    def known_n_current_infected(self):
        return len(self.current_infected_nodes)

    @property
    def isolated_nodes(self) -> List[int]:
        # TODO: Assuming these are always known for now, as only the result of agent action
        if self._isolated_nodes is None:
            self._isolated_nodes = self.graph.current_isolated_nodes
        return self._isolated_nodes

    @property
    def unknown_nodes(self) -> List[int]:
        """Unknown nodes, excludes dead (as these are always known)"""
        if self._unknown_nodes is None:
            self._unknown_nodes = [nk for nk, nv in self.graph.g_.nodes.data() if nv["status"].clear is None]
        return self._unknown_nodes

    @property
    def current_alive_nodes(self) -> List[int]:
        """Status known, excludes dead"""
        if self._known_nodes is None:
            if self.test_rate >= 1:
                self._known_nodes = self.graph.current_alive_nodes
            else:
                self._known_nodes = [nk for nk, nv in self.graph.g_.nodes.data()
                                     if nv["status"].alive]
        return self._known_nodes

    @property
    def current_infected_nodes(self) -> List[int]:
        if self._current_infected_nodes is None:
            if self.test_rate >= 1:
                self._current_infected_nodes = self.graph.current_infected_nodes
            else:
                self._current_infected_nodes = [nk for nk, nv in self.graph.g_.nodes.data()
                                                if nv["status"].infected]
        return self._current_infected_nodes

    @property
    def current_immune_nodes(self) -> List[int]:
        if self._current_immune_nodes is None:
            if self.test_rate >= 1:
                self._current_immune_nodes = self.graph.current_immune_nodes
            else:
                self._current_immune_nodes = [nk for nk, nv in self.graph.g_.nodes.data()
                                              if nv["status"].immune]
        return self._current_immune_nodes

    @property
    def current_clear_nodes(self) -> List[int]:
        if self._current_clear_nodes is None:
            if self.test_rate >= 1:
                self._current_clear_nodes = self.graph.current_clear_nodes
            else:
                self._current_clear_nodes = [nk for nk, nv in self.graph.g_.nodes.data()
                                             if nv["status"].clear]
        return self._current_clear_nodes

    def test_population(self, time_step: int) -> None:
        """
        Test random members of the environment, based on testing rate.

        Chance of testing infected is grater than testing asymptomatic.

        Already identified infected nodes get a free test each turn.
        """
        clear_test_rate = self.test_rate / 2
        infected_test_rate = self.test_rate * 2

        if clear_test_rate > 1:
            clear_test_rate = 1
        if infected_test_rate > 1:
            infected_test_rate = 1

        for n in self.graph.current_clear_nodes:
            if self._random_state.binomial(1, clear_test_rate):
                self.graph.g_.nodes[n]['status'].last_tested = time_step

        for n in self.graph.current_infected_nodes:
            if self._random_state.binomial(1, infected_test_rate):
                self.graph.g_.nodes[n]['status'].last_tested = time_step

        for n in self.current_infected_nodes:
            self.graph.g_.nodes[n]['status'].last_tested = time_step

    def update_observed_statuses(self, time_step: int) -> int:
        known_new_infections = 0

        for nk, nv in self.graph.g_.nodes.data():
            # Is dead
            if not nv['alive']:
                nv['status'] = Status(alive=False)
                continue

            # Is infected and tested this turn
            if (nv['infected'] > 0) and (nv["status"].last_tested == time_step):
                nv['status'].infected = True
                known_new_infections += 1

            # Is clear or immune and was tested this turn
            if nv['infected'] == 0 and (nv["status"].last_tested == time_step):
                nv['status'].recovered = True
                if nv['immune'] >= self.graph.considered_immune_threshold:
                    nv['status'].immune = True

            # Test has expired (only for clear and immune nodes)
            if ((nv["status"].clear or nv["status"].immune)
                    and ((time_step - nv["status"].last_tested) > self.test_validity_period)):
                nv['status'].set_health_unknown()

        return known_new_infections

    def plot(self,
             ax: Union[None, plt.Axes] = None,
             colours: Dict[str, str] = None) -> None:
        """
        Plot the observed network graph.

        TODO: Duplicates functionality with Graph.plot...

        :param ax: Matplotlib to draw on.
        :param colours: Colours to use for plots, for example, from defaults set in history, if available.
        """
        sns.set()

        if colours is None:
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
                               nodelist=self.current_clear_nodes,
                               node_color=colours.get('Known current clear', '#1f77b4'),
                               node_size=10,
                               ax=ax)
        nx.draw_networkx_nodes(self.graph.g_, self.graph.g_pos_,
                               nodelist=self.current_immune_nodes,
                               node_color=colours.get('Known total immune', '#9467bd'),
                               node_size=10,
                               ax=ax)
        nx.draw_networkx_nodes(self.graph.g_, self.graph.g_pos_,
                               nodelist=self.current_infected_nodes,
                               node_color=colours.get('Known current infections', '#d62728'),
                               node_size=10,
                               ax=ax)
        nx.draw_networkx_nodes(self.graph.g_, self.graph.g_pos_,
                               nodelist=self.graph.current_dead_nodes,
                               node_color=colours.get('Total deaths', 'k'),
                               node_size=10,
                               ax=ax)
        nx.draw_networkx_edges(self.graph.g_, self.graph.g_pos_,
                               width=1 / (self.graph.total_population / 5),
                               ax=ax)

    def clone(self) -> "ObservationSpace":
        """Clone a fresh object with same seed (could be None)."""
        return ObservationSpace(graph=self.graph.clone(), test_rate=self.test_rate,
                                test_validity_period=self.test_validity_period, seed=self.seed)


if __name__ == "__main__":
    obs = ObservationSpace(Graph())

    obs.plot_matrix()
    obs.plot_summary()
