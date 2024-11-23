from dataclasses import dataclass
from typing import Dict, List, Tuple, Union

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
        self._masked_nodes: Union[List[int], None] = None

    def _attach_status_to_graph(self):
        for _, nv in self.graph.g_.nodes.data():
            nv["status"] = Status()

    def _prepare_random_state(self) -> None:
        self._random_state = np.random.RandomState(seed=self.seed)

    def state_summary(self) -> np.ndarray:
        """
        Vector representing n of each node type. Contains known values.

        Note this matches order returned by Status.state + unknown nodes at end at [-1].
        """
        return np.array(
            [
                len(self.current_alive_nodes),
                len(self.current_clear_nodes),
                len(self.current_infected_nodes),
                len(self.current_immune_nodes),
                len(self.current_isolated_nodes),
                len(self.current_masked_nodes),
                len(self.unknown_nodes),
            ]
        )

    def state_graph(self) -> np.ndarray:
        """Node x node matrix representing graph. All connections are known, so same as .graph."""
        return self.graph.state_graph()

    def state_nodes(self) -> np.ndarray:
        """Node x node_state matrix. Uses node["status"] which handles known node state."""
        return np.array([nd["status"].state for _, nd in self.graph.g_.nodes.data()])

    def state_full(self) -> np.ndarray:
        """.state_nodes + .state_graph"""
        return np.concatenate([self.state_nodes(), self.state_graph()], axis=1)

    @property
    def state(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (self.state_summary(), self.state_graph(), self.state_nodes())

    def plot_matrix(self, ax: Union[None, plt.Axes] = None) -> plt.Figure:
        fig = sns.heatmap(self.state_graph(), ax=ax)

        return fig

    def plot_summary(self, ax: Union[None, plt.Axes] = None) -> plt.Axes:

        ax_ = ax
        for i, c in enumerate(Status().state_features_names):
            ax_ = sns.distplot(self.state_nodes()[:, i], label=c, kde=False, ax=ax)
        plt.legend()

        return ax_

    @property
    def known_n_current_infected(self):
        return len(self.current_infected_nodes)

    @property
    def current_masked_nodes(self) -> List[int]:
        if self._masked_nodes is None:
            self._masked_nodes = self.graph.current_masked_nodes
        return self._masked_nodes

    @property
    def current_isolated_nodes(self) -> List[int]:
        # TODO: Assuming these are always known for now, as only the result of agent action
        if self._isolated_nodes is None:
            self._isolated_nodes = self.graph.current_isolated_nodes
        return self._isolated_nodes

    @property
    def unknown_nodes(self) -> List[int]:
        """Unknown nodes, excludes dead (as these are always known)"""
        if self._unknown_nodes is None:
            self._unknown_nodes = [
                nk
                for nk, nv in self.graph.g_.nodes.data()
                if nv["status"].clear is None
            ]
        return self._unknown_nodes

    @property
    def current_alive_nodes(self) -> List[int]:
        """Status known, excludes dead"""
        if self._known_nodes is None:
            if self.test_rate >= 1:
                self._known_nodes = self.graph.current_alive_nodes
            else:
                self._known_nodes = [
                    nk for nk, nv in self.graph.g_.nodes.data() if nv["status"].alive
                ]
        return self._known_nodes

    @property
    def current_infected_nodes(self) -> List[int]:
        if self._current_infected_nodes is None:
            if self.test_rate >= 1:
                self._current_infected_nodes = self.graph.current_infected_nodes
            else:
                self._current_infected_nodes = [
                    nk for nk, nv in self.graph.g_.nodes.data() if nv["status"].infected
                ]
        return self._current_infected_nodes

    @property
    def current_immune_nodes(self) -> List[int]:
        if self._current_immune_nodes is None:
            if self.test_rate >= 1:
                self._current_immune_nodes = self.graph.current_immune_nodes
            else:
                self._current_immune_nodes = [
                    nk for nk, nv in self.graph.g_.nodes.data() if nv["status"].immune
                ]
        return self._current_immune_nodes

    @property
    def current_clear_nodes(self) -> List[int]:
        if self._current_clear_nodes is None:
            if self.test_rate >= 1:
                self._current_clear_nodes = self.graph.current_clear_nodes
            else:
                self._current_clear_nodes = [
                    nk for nk, nv in self.graph.g_.nodes.data() if nv["status"].clear
                ]
        return self._current_clear_nodes

    def test_population(self, time_step: int) -> None:
        """
        Test random members of the environments, based on testing rate.

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
                self.graph.g_.nodes[n]["status"].last_tested = time_step

        for n in self.graph.current_infected_nodes:
            if self._random_state.binomial(1, infected_test_rate):
                self.graph.g_.nodes[n]["status"].last_tested = time_step

        for n in self.current_infected_nodes:
            self.graph.g_.nodes[n]["status"].last_tested = time_step

    def update_observed_statuses(self, time_step: int) -> int:
        known_new_infections = 0

        for nk, nv in self.graph.g_.nodes.data():
            # Is dead
            if not nv["alive"]:
                nv["status"] = Status(alive=False)
                continue

            # Update if tested this turn
            if nv["status"].last_tested == time_step:
                # Has mask
                if nv["mask"] > 0:
                    nv["status"].masked = True
                else:
                    nv["status"].masked = False

                # Is infected
                if nv["infected"] > 0:
                    nv["status"].infected = True
                    known_new_infections += 1

                # Is clear or immune
                if nv["infected"] == 0:
                    nv["status"].recovered = True
                    if nv["immune"] >= self.graph.considered_immune_threshold:
                        nv["status"].immune = True

            # Test has expired (only for clear and immune nodes)
            if (nv["status"].clear or nv["status"].immune) and (
                (time_step - nv["status"].last_tested) > self.test_validity_period
            ):
                nv["status"].set_health_unknown()

        return known_new_infections

    def plot(
        self,
        ax: Union[None, plt.Axes] = None,
        colours: Dict[str, str] = None,
        god_mode: bool = True,
    ) -> None:
        """
        Plot the full or observed network graph.

        :param ax: Matplotlib to draw on.
        :param colours: Colours to use for plots, for example, from defaults set in history, if available.
        :param god_mode: If True, plot by-pass observation space filters and plot the whole environments state. If False,
                         plot the observable graph only.
        """
        sns.set()

        if colours is None:
            # Will use defaults defined along with data below
            colours = {}

        # Position nodes, but only if they haven't been before (this is expensive and nodes changing location is
        # confusing)
        if self.graph.g_pos_ is None:
            self.graph.g_pos_ = nx.spring_layout(self.graph.g_, seed=self.seed)

        common_plotting_args = {
            "G": self.graph.g_,
            "pos": self.graph.g_pos_,
            "ax": ax,
            "node_size": 12,
            "linewidths": 2,
            "alpha": 0.75,
        }

        if god_mode:
            info_source = self.graph
        else:
            info_source = self
            nx.draw_networkx_nodes(
                **common_plotting_args,
                nodelist=self.unknown_nodes,
                node_color=colours.get("Unknown", "#bdbcbb")
            )

        # Draw selected set of properties and include default colours in case not defined in colours
        for pk, pv in {
            "Known current clear": (info_source.current_clear_nodes, "#1f77b4"),
            "Known total immune": (info_source.current_immune_nodes, "#9467bd"),
            "Known current infections": (info_source.current_infected_nodes, "#d62728"),
            "Total deaths": (self.graph.current_dead_nodes, "k"),
        }.items():
            nx.draw_networkx_nodes(
                **common_plotting_args,
                nodelist=pv[0],
                node_color=colours.get(pk, pv[1])
            )

        # Draw mask indicators and edges
        nx.draw_networkx_nodes(
            self.graph.g_,
            self.graph.g_pos_,
            ax=ax,
            nodelist=info_source.current_masked_nodes,
            node_color="k",
            node_size=2,
            alpha=1,
            linewidths=6,
        )
        nx.draw_networkx_edges(
            self.graph.g_,
            self.graph.g_pos_,
            ax=ax,
            width=1 / (self.graph.total_population / 5),
        )

    def clone(self) -> "ObservationSpace":
        """Clone a fresh object with same seed (could be None)."""
        return ObservationSpace(
            graph=self.graph.clone(),
            test_rate=self.test_rate,
            test_validity_period=self.test_validity_period,
            seed=self.seed,
        )


if __name__ == "__main__":
    obs = ObservationSpace(Graph())

    obs.plot_matrix()
    obs.plot_summary()
