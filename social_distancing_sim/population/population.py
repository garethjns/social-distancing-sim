"""
TODO:
  - Split out observation space from population.
  - Split healthcare into separate object?
"""

import glob
import os
import shutil
from dataclasses import dataclass
from typing import Union, Dict, List, Tuple

import imageio
import networkx as nx
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from social_distancing_sim.disease.disease import Disease
from social_distancing_sim.population.history import History

sns.set()


@dataclass
class Population:
    disease: Disease
    name: str = "unnamed_population"
    seed: int = None

    community_n: int = 5
    community_size_mean: int = 5
    community_size_std: int = 1
    community_p_in: float = 0.2
    community_p_out: float = 0.1

    healthcare_capacity: float = 0.15
    healthcare_efficiency: float = None
    healthcare_test_rate: float = 1
    healthcare_test_validity_period: float = 5

    def __post_init__(self) -> None:
        self._prepare_random_state()

        self._community_sizes: np.ndarray = self.state.poisson(self.state.normal(size=self.community_n)
                                                               * self.community_size_std
                                                               + self.community_size_mean)
        self.g_: nx.classes.graph.Graph = None
        self.g_pos_: Union[None, Dict[int, np.ndarray]] = None
        self._generate_graph()

        self.total_population = len(self.g_.nodes)
        self._step: int = 0
        self.history: History[str, List[int]] = History.with_defaults()

        self._prepare_output_path()
        self._prepare_figure()

    def _prepare_random_state(self) -> None:
        self.state = np.random.RandomState(seed=self.seed)

    def _prepare_figure(self) -> None:
        plt.close()
        fig = plt.figure()
        gs = fig.add_gridspec(6, 1)

        graph_ax = fig.add_subplot(gs[:4, 0])
        ts_ax = fig.add_subplot(gs[4:, 0])

        self._figure = fig
        self._graph_ax = graph_ax
        self._ts_ax = ts_ax

    def _prepare_output_path(self) -> None:
        self.output_path = f"{self.name}"
        shutil.rmtree(self.output_path,
                      ignore_errors=True)

        self.graph_path = f"{self.output_path}/graphs/"
        os.makedirs(self.graph_path,
                    exist_ok=True)

    @property
    def known_n_current_infected(self):
        return len(self.known_current_infected_nodes)

    @property
    def unknown_nodes(self) -> List[int]:
        return [nk for nk, nv in self.g_.nodes.data() if (nv.get("status", '') == '')]

    @property
    def known_current_infected_nodes(self) -> List[int]:
        if self.healthcare_test_rate >= 1:
            return self.current_infected_nodes
        else:
            return [nk for nk, nv in self.g_.nodes.data() if (nv.get("status", '') == 'infected')]

    @property
    def known_current_immune_nodes(self) -> List[int]:
        if self.healthcare_test_rate >= 1:
            return self.current_immune_nodes
        else:
            return [nk for nk, nv in self.g_.nodes.data() if (nv.get("status", '') == 'immune')]

    @property
    def known_current_clear_nodes(self) -> List[int]:
        if self.healthcare_test_rate >= 1:
            return self.current_clear_nodes
        else:
            return [nk for nk, nv in self.g_.nodes.data() if (nv.get("status", '') == 'clear')]

    @property
    def known_current_alive_nodes(self) -> List[int]:
        """We know who's alive without testing...."""
        return self.current_alive_nodes

    @property
    def known_current_dead_nodes(self) -> List[int]:
        """We know who's dead without testing...."""
        return self.current_dead_nodes

    @property
    def n_current_infected(self) -> int:
        return len(self.current_infected_nodes)

    @property
    def current_infected_nodes(self) -> List[int]:
        return [nk for nk, nv in self.g_.nodes.data() if (nv["infected"] > 0) & nv["alive"]]

    @property
    def current_immune_nodes(self) -> List[int]:
        return [nk for nk, nv in self.g_.nodes.data() if nv["immune"] & nv["alive"]]

    @property
    def current_clear_nodes(self) -> List[int]:
        return [nk for nk, nv in self.g_.nodes.data() if (nv["infected"] == 0) & nv["alive"]]

    @property
    def current_alive_nodes(self) -> List[int]:
        return [nk for nk, nv in self.g_.nodes.data() if nv["alive"]]

    @property
    def current_dead_nodes(self) -> List[int]:
        return [nk for nk, nv in self.g_.nodes.data() if not nv["alive"]]

    @property
    def overall_death_rate(self):
        if len(self.current_dead_nodes) > 0:
            death_rate = len(self.current_dead_nodes) / self.total_population
        else:
            death_rate = 0

        return death_rate

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

    def plot_graph(self) -> None:
        """Plot the network graph."""
        if self.g_pos_ is None:
            self.g_pos_ = nx.spring_layout(self.g_,
                                           seed=self.seed)

        nx.draw_networkx_nodes(self.g_, self.g_pos_,
                               nodelist=self.unknown_nodes,
                               node_color=self.history.colours.get('Unknown', '#bdbcbb'),
                               node_size=10,
                               ax=self._graph_ax)
        nx.draw_networkx_nodes(self.g_, self.g_pos_,
                               nodelist=self.known_current_clear_nodes,
                               node_color=self.history.colours.get('Current clear', '#1f77b4'),
                               node_size=10,
                               ax=self._graph_ax)
        nx.draw_networkx_nodes(self.g_, self.g_pos_,
                               nodelist=self.known_current_immune_nodes,
                               node_color=self.history.colours.get('Total immune', '#9467bd'),
                               node_size=10,
                               ax=self._graph_ax)
        nx.draw_networkx_nodes(self.g_, self.g_pos_,
                               nodelist=self.known_current_infected_nodes,
                               node_color=self.history.colours.get('Current infections', '#d62728'),
                               node_size=10,
                               ax=self._graph_ax)
        nx.draw_networkx_nodes(self.g_, self.g_pos_,
                               nodelist=self.known_current_dead_nodes,
                               node_color=self.history.colours.get('Total deaths', 'k'),
                               node_size=10,
                               ax=self._graph_ax)
        nx.draw_networkx_edges(self.g_, self.g_pos_,
                               width=0.01,
                               ax=self._graph_ax)

    def _infect_random(self) -> None:
        """Infect a random node."""
        self.g_.nodes[self.state.randint(0, len(self.g_))]['infected'] = 1

    def _infect_neighbours(self) -> int:
        """For all the currently infected nodes, attempt to infect neighbours."""
        new_infections = 0
        for n in self.current_infected_nodes:
            # Get neighbours
            for nb in self.g_.neighbors(n):
                node = self.g_.nodes[nb]
                if (not node["immune"]) and (not node["infected"] > 0):
                    node["infected"] = self.state.binomial(1, self.disease.virulence)

                    new_infections += node["infected"]

        return new_infections

    def _modified_recovery_rate(self) -> float:
        """
        Penalise the disease survivability by current healthcare burden.

        If utilisation is below capacity, no penalty. If above, reduce survivability by some function of this.
        Limited to minimum of 50% normal recovery rate.
        """
        # Set current healthcare penalty
        if self.n_current_infected < (self.total_population * self.healthcare_capacity):
            # No penalty
            current_healthcare_penalty = 0
        else:
            current_healthcare_penalty = (self.n_current_infected / (self.healthcare_capacity
                                                                     * self.total_population) - 1) / 8
        # Reduce disease survivability by this %
        return max(self.disease.recovery_rate - self.disease.recovery_rate * current_healthcare_penalty,
                   self.disease.recovery_rate * 0.5)

    def _conclude(self) -> Tuple[int, int]:
        """
        For all the currently infected nodes, see if it's possible to conclude the disease.

        The chance of a conclusion increases with the duration of the disease, and the outcome (survive or die) is
        modified by the recovery rate of the disease and the current healthcare burden.
        """

        deaths = 0
        recoveries = 0
        for n in self.current_infected_nodes:
            node = self.g_.nodes[n]
            # Decide end of disease
            if node["infected"] > self.state.normal(self.disease.duration_mean,
                                                    self.disease.duration_std,
                                                    size=1):
                node["infected"] = 0
                node["immune"] = True

                if self.state.binomial(1, self._modified_recovery_rate()):
                    node["alive"] = True
                    recoveries += 1
                else:
                    node["alive"] = False
                    deaths += 1

            else:
                # Continue disease progression
                node["infected"] += 1

        return deaths, recoveries

    def _log(self, new_infections: int, deaths: int, recoveries: int) -> None:
        self.history["Current infections"].append(self.known_n_current_infected)
        self.history["Current clear"].append(self.total_population - self.known_n_current_infected)
        self.history["Current recovery rate"].append(self._modified_recovery_rate())
        self.history["Number alive"].append(len(self.known_current_alive_nodes))
        self.history["Total deaths"].append(len(self.known_current_dead_nodes))
        self.history["Total immune"].append(len(self.known_current_immune_nodes))
        self.history["New infections"].append(new_infections)
        self.history["New deaths"].append(deaths)
        self.history["Total recovered"].append(recoveries)
        self.history["graph"].append(self.g_.copy())
        self.history["Total infections"].append(np.cumsum(self.history["New infections"]))

        # Stats
        self.history["Current infection prop"].append(self.history["Current infections"][-1] /
                                                      self.total_population)
        self.history["Overall infection prop"].append(self.history["Total infections"][-1] /
                                                      self.total_population)
        self.history["Current death prop"].append(self.history["New deaths"][-1] /
                                                  self.total_population)
        self.history["Overall death prop"].append(self.history["Total deaths"][-1] / self.total_population)
        self.history["Overall Infected death rate"].append(self.history["Total deaths"][-1]
                                                           / self.history["Total infections"][-1])

    def replay(self, duration: float = 1) -> str:
        """

        :param duration: Frame duration,
        :return: Path to rendered gif.
        """
        # Find all previously saved steps
        fns = glob.glob(f"{self.graph_path}*_graph.png")
        # Ensure ordering
        fns = [f.replace('\\', '/') for f in fns]
        sorted_idx = np.argsort([int(f.split('_graph.png')[0].split(self.graph_path)[1]) for f in fns])
        fns = np.array(fns)[sorted_idx]

        # Generate gif
        output_path = f"{self.output_path}/replay.gif"
        images = [imageio.imread(f) for f in fns]
        imageio.mimsave(output_path, images,
                        duration=duration,
                        subrectangles=True)

        return output_path

    def plot_ts(self) -> None:
        self.history.plot(["Current infections", "Total immune", "Total deaths"],
                          ax=self._ts_ax,
                          show=False)
        cap = self.healthcare_capacity * self.total_population
        self._ts_ax.plot([0, self._step], [cap, cap],
                         linestyle="--",
                         color='k')

    def plot(self, save: bool = True, show: bool = True) -> None:
        self._prepare_figure()
        self.plot_graph()
        self.plot_ts()

        self._graph_ax.set_title(f"{self.name}, day {self._step}: "
                                 f"Deaths = {len(self.current_dead_nodes)}")

        if save:
            plt.savefig(f"{self.output_path}/graphs/{self._step}_graph.png")

        if show:
            plt.show()

    def _test_population(self):
        """
        Test random members of the population, based on testing rate.

        Chance of testing infected is grater than testing asymptomatic.
        """
        clear_test_rate = self.healthcare_test_rate / 2
        infected_test_rate = self.healthcare_test_rate * 2

        if clear_test_rate > 1:
            clear_test_rate = 1
        if infected_test_rate > 1:
            infected_test_rate = 1

        for n in self.current_clear_nodes:
            if self.state.binomial(1, clear_test_rate):
                self.g_.nodes[n]['last_tested'] = self._step

        for n in self.current_infected_nodes:
            if self.state.binomial(1, infected_test_rate):
                self.g_.nodes[n]['last_tested'] = self._step

    def _update_observed_statuses(self) -> int:
        known_new_infections = 0

        for nk, nv in self.g_.nodes.data():
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
            if nv['infected'] > 0 and (nv.get("last_tested", -1) == self._step):
                nv['status'] = 'infected'
                known_new_infections += 1

            # Is clear and tested this turn
            if nv['infected'] == 0 and (nv.get("last_tested", -1) == self._step):
                nv['status'] = 'clear'

            # Test has expired (only for clear nodes)
            if (nv.get("status", "clear")
                    and ((self._step - nv.get("last_tested", 0)) > self.healthcare_test_validity_period)):
                nv['status'] = ''

        return known_new_infections

    def step(self,
             plot: bool = True,
             save: bool = True) -> None:
        if self._step == 0:
            self._infect_random()

        new_infections = self._infect_neighbours()
        deaths, recoveries = self._conclude()
        self._test_population()
        known_new_infections = self._update_observed_statuses()

        self._log(known_new_infections, deaths, recoveries)

        if plot or save:
            print(f"Step {self._step} concluded")
            self.plot(show=plot,
                      save=save)

        self._step += 1

    def run(self, steps: 10,
            plot: bool = True,
            save: bool = True) -> None:
        """
        Run simulation for a number of iterations.

        :param steps: Number of steps to run.
        :param plot: Display plots while running.
        :param save: Save plot of each step while running.
        """
        for _ in range(steps):
            self.step(plot=plot,
                      save=save)
