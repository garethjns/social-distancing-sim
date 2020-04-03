import glob
import os
import shutil
from dataclasses import dataclass
from typing import List, Tuple, Union

import imageio
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from social_distancing_sim.disease.disease import Disease
from social_distancing_sim.population.healthcare import Healthcare
from social_distancing_sim.population.history import History
from social_distancing_sim.population.observation_space import ObservationSpace
import time


@dataclass
class Population:
    disease: Disease
    healthcare: Healthcare
    observation_space: ObservationSpace
    name: str = "unnamed_population"
    seed: Union[None, int] = None

    def __post_init__(self) -> None:
        self._prepare_random_state()

        self.total_population = self.observation_space.graph.total_population
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

    def _infect_random(self) -> None:
        """Infect a random node."""
        self.disease.force_infect(self.observation_space.graph.g_.nodes[self.state.randint(0, self.total_population)])

    def _infect_neighbours(self) -> int:
        """
        For all the currently infected nodes, attempt to infect neighbours.

        TODO: This is the biggest time sink
        """
        new_infections = 0
        for n in self.observation_space.graph.current_infected_nodes:
            # Get neighbours
            for nb in self.observation_space.graph.g_.neighbors(n):
                node = self.disease.try_to_infect(self.observation_space.graph.g_.nodes[nb])

                if node["infected"] == 1:
                    # new infection, count
                    new_infections += node["infected"]

        return new_infections

    def _conclude_all(self) -> Tuple[int, int]:
        """
        For all the currently infected nodes, see if it's possible to conclude the disease.

        The chance of a conclusion increases with the duration of the disease, and the outcome (survive or die) is
        modified by the recovery rate of the disease and the current healthcare burden.
        """

        deaths = 0
        recoveries = 0
        recovery_rate_modifier = self.healthcare.recovery_rate_penalty(
            n_current_infected=self.observation_space.graph.n_current_infected)
        for n in self.observation_space.graph.current_infected_nodes:
            node = self.disease.conclude(self.observation_space.graph.g_.nodes[n],
                                         recovery_rate_modifier=recovery_rate_modifier)

            # Outcome is either recovery, death, or continuation
            if node["alive"]:
                if node["infected"] == 0:
                    recoveries += 1
            else:
                deaths += 1

        return deaths, recoveries

    def _log(self, new_infections: int, known_new_infections: int, deaths: int, recoveries: int) -> None:
        """Log full space and observed space."""
        self.history.log({"Current infections": self.observation_space.graph.n_current_infected,
                          "Known current infections": self.observation_space.known_n_current_infected,
                          "Current clear": self.total_population - self.observation_space.graph.n_current_infected,
                          "Known current clear": (self.total_population
                                                  - self.observation_space.known_n_current_infected),
                          "Current recovery rate penalty": self.healthcare.recovery_rate_penalty(
                              self.observation_space.graph.n_current_infected),
                          "Number alive": len(self.observation_space.graph.current_alive_nodes),
                          "Total deaths": len(self.observation_space.graph.current_dead_nodes),
                          "Total immune": len(self.observation_space.graph.current_immune_nodes),
                          "Known total immune": len(self.observation_space.known_current_immune_nodes),
                          "New infections": new_infections,
                          "Known new infections": known_new_infections,
                          "New deaths": deaths,
                          "Total recovered": recoveries,
                          "Total infections": np.cumsum(self.history["New infections"]),
                          "Known total infections": np.cumsum(self.history["Known new infections"])})

        # Dependent
        self.history.log({
            "Current infection prop": self.history["Current infections"][-1] / self.total_population,
            "Known current infection prop": (self.history["Known current infections"][-1]
                                             / self.total_population),
            "Overall infection prop": self.history["Total infections"][-1] / self.total_population,
            "Known overall infection prop": (self.history["Known total infections"][-1]
                                             / self.total_population),
            "Current death prop": self.history["New deaths"][-1] / self.total_population,
            "Overall death prop": self.history["Total deaths"][-1] / self.total_population,
            "Overall Infected death rate": (self.history["Total deaths"][-1]
                                            / self.history["Total infections"][-1]),
            "Known overall Infected death rate": (self.history["Total deaths"][-1]
                                                  / self.history["Total infections"][-1])})

    def replay(self, duration: float = 0.3) -> str:
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
        self._ts_ax.plot([0, self._step], [self.healthcare.capacity, self.healthcare.capacity],
                         linestyle="--",
                         color='k')

    def plot(self, save: bool = True, show: bool = True) -> None:
        sns.set()

        self._prepare_figure()
        self.observation_space.plot(ax=self._graph_ax)
        self.plot_ts()

        self._graph_ax.set_title(f"{self.name}, day {self._step}: "
                                 f"Deaths = {len(self.observation_space.graph.current_dead_nodes)}")

        if save:
            plt.savefig(f"{self.output_path}/graphs/{self._step}_graph.png")

        if show:
            plt.show()

    def step(self,
             plot: bool = True,
             save: bool = True) -> None:
        if self._step == 0:
            self._infect_random()

        _ = self._infect_neighbours()
        deaths, recoveries = self._conclude_all()
        self.observation_space.test_population(self._step)
        known_new_infections = self.observation_space.update_observed_statuses(self._step)

        self._log(new_infections=known_new_infections, known_new_infections=known_new_infections,
                  deaths=deaths, recoveries=recoveries)

        if plot or save:
            print(f"Step {self._step} concluded")
            self.plot(show=plot,
                      save=save)

        self.observation_space.graph.reset_cached_values()
        self.observation_space.reset_cached_values()

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
        t0 = time.time()
        for _ in range(steps):
            self.step(plot=plot,
                      save=save)

        print(f"Ran {steps} steps in {np.round(time.time() - t0, 2)}s")
