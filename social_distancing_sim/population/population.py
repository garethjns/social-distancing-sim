import glob
import os
import shutil
import time
import warnings
from dataclasses import dataclass
from typing import List, Tuple, Union

import imageio
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from tqdm import tqdm

from social_distancing_sim.disease.disease import Disease
from social_distancing_sim.population.healthcare import Healthcare
from social_distancing_sim.population.history import History
from social_distancing_sim.population.observation_space import ObservationSpace
from social_distancing_sim.population.scoring import Scoring


@dataclass
class Population:
    observation_space: ObservationSpace
    disease: Disease = Disease()
    healthcare: Healthcare = Healthcare()
    scoring: Scoring = Scoring()
    name: str = "unnamed_population"
    seed: Union[None, int] = None

    random_infection_chance: float = 0.01

    plot_both: bool = True
    plot_ts_fields_g1: List[str] = None
    plot_ts_fields_g2: List[str] = None
    plot_ts_obs_fields_g1: List[str] = None
    plot_ts_obs_fields_g2: List[str] = None

    def __post_init__(self) -> None:
        self._prepare_random_state()

        self.total_population = self.observation_space.graph.total_population
        self._step: int = 0
        self.history: History[str, List[int]] = History.with_defaults()
        self._total_steps: int = 0
        self.output_path: Union[str, None] = None

        sns.set()

    def _prepare_random_state(self) -> None:
        self.state = np.random.RandomState(seed=self.seed)

    def _prepare_figure(self) -> None:
        plt.close()

        self._g2_on = False
        ts_ax_g2 = None

        if self.plot_ts_fields_g1 is None:
            self.plot_ts_fields_g1 = ["Current infections", "Total immune", "Total deaths"]
        if self.plot_ts_obs_fields_g1 is None:
            self.plot_ts_obs_fields_g1 = ["Known current infections", "Known total immune", "Total deaths"]
        if self.plot_ts_fields_g2 is None:
            self.plot_ts_fields_g2 = []
        if self.plot_ts_obs_fields_g2 is None:
            self.plot_ts_obs_fields_g2 = []

        if len(self.plot_ts_fields_g2) > 0:
            self._g2_on = True

        height = 5
        nrows = 6
        if self._g2_on:
            height += height / nrows * 2
            nrows += 2

        if (self.observation_space.test_rate < 1) & self.plot_both:
            # Plot reality and observed space separately
            fig = plt.figure(figsize=(height * 2, height))
            gs = fig.add_gridspec(nrows, 2)
            graph_ax = [fig.add_subplot(gs[:4, 0]), fig.add_subplot(gs[:4, 1])]
            ts_ax_g1 = [fig.add_subplot(gs[4:6, 0]), fig.add_subplot(gs[4:6, 1])]
            if self._g2_on:
                ts_ax_g2 = [fig.add_subplot(gs[6:8, 0]), fig.add_subplot(gs[6:8, 1])]
        else:
            # Observed is reality, just plot single figure
            fig = plt.figure(figsize=(6.4, height))
            gs = fig.add_gridspec(nrows, 1)
            graph_ax = [fig.add_subplot(gs[:4, 0])]
            ts_ax_g1 = [fig.add_subplot(gs[4:6, 0])]
            if self._g2_on:
                ts_ax_g2 = [fig.add_subplot(gs[6:8, 0])]

        self._figure = fig
        self._graph_ax: List[plt.Axes] = graph_ax
        self._ts_ax_g1: List[plt.Axes] = ts_ax_g1
        self._ts_ax_g2: List[plt.Axes] = ts_ax_g2

    def _prepare_output_path(self) -> None:
        if self.output_path is None:
            self.output_path = f"{self.name}"
            shutil.rmtree(self.output_path,
                          ignore_errors=True)

            self.graph_path = f"{self.output_path}/graphs/"
            os.makedirs(self.graph_path,
                        exist_ok=True)

    def _infect_random(self) -> None:
        """Infect a random node."""
        node_id = self.observation_space.graph.current_clear_nodes[
            self.state.randint(0, len(self.observation_space.graph.current_clear_nodes))]
        self.disease.force_infect(self.observation_space.graph.g_.nodes[node_id])

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

    def _log(self, new_infections: int, known_new_infections: int, deaths: int, recoveries: int,
             score: float = 0.0,
             obs_score: float = 0.0) -> None:
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
                          "Mean immunity (of immune nodes)": np.mean(
                              [self.observation_space.graph.g_.nodes[n]["immune"]
                               for n in self.observation_space.graph.current_immune_nodes]),
                          "Mean immunity (of all alive nodes)": np.mean(
                              [self.observation_space.graph.g_.nodes[n].get("immune", 0)
                               for n in
                               self.observation_space.graph.current_alive_nodes]),
                          "Known total immune": len(
                              self.observation_space.current_immune_nodes),
                          "Known mean immunity (of immune nodes)": np.mean(
                              [self.observation_space.graph.g_.nodes[n]["immune"]
                               for n in self.observation_space.current_immune_nodes]),
                          "Known mean immunity (of all alive nodes)": np.mean(
                              [self.observation_space.graph.g_.nodes[n].get("immune", 0)
                               for n in
                               self.observation_space.current_alive_nodes]),
                          "New infections": new_infections,
                          "Known new infections": known_new_infections,
                          "New deaths": deaths,
                          "Total recovered": recoveries,
                          "Total infections": np.sum(self.history["New infections"]),
                          "Known total infections": np.sum(self.history["Known new infections"]),
                          "Score": score,
                          "Observed score": obs_score})

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

    def replay(self, duration: float = 0.2) -> str:
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
        for ax, fields in zip(self._ts_ax_g1, [self.plot_ts_fields_g1, self.plot_ts_obs_fields_g1]):
            self.history.plot(ks=fields,
                              x_lim=(-1, self._total_steps),
                              y_lim=(-10, int(self.total_population + self.total_population * 0.05)),
                              x_label='Day' if not self._g2_on else None,
                              remove_x_tick_labels=self._g2_on,
                              ax=ax,
                              show=False)
            ax.plot([0, self._step], [self.healthcare.capacity, self.healthcare.capacity],
                    linestyle="--",
                    color='k')

        if self._g2_on:
            for ax, fields in zip(self._ts_ax_g2, [self.plot_ts_fields_g2, self.plot_ts_obs_fields_g2]):
                self.history.plot(ks=fields,
                                  y_label='Score',
                                  x_lim=(-1, self._total_steps),
                                  # y_lim=(-0.1, 1.1),
                                  ax=ax,
                                  show=False)

    def plot_graphs(self):
        title = f"{self.name}, day {self._step} (deaths = {len(self.observation_space.graph.current_dead_nodes)})"

        self.observation_space.graph.plot(ax=self._graph_ax[0])
        self._graph_ax[0].set_title(f"Full sim: {title}")

        if (self.observation_space.test_rate < 1) & self.plot_both:
            self.observation_space.plot(ax=self._graph_ax[1])
            self._graph_ax[1].set_title(f"Observed: {title}")

    def plot(self, save: bool = True, show: bool = True) -> None:

        self._prepare_figure()
        self.plot_graphs()
        self.plot_ts()

        self._figure.tight_layout()

        if save:
            self._prepare_output_path()
            plt.savefig(f"{self.output_path}/graphs/{self._step}_graph.png")

        if show:
            plt.show()

    def _update_immunities(self):
        for node in self.observation_space.graph.current_immune_nodes:
            self.disease.decay_immunity(self.observation_space.graph.g_.nodes[node])

    def step(self) -> None:
        self.observation_space.reset_cached_values()
        self.observation_space.graph.reset_cached_values()

        if (self._step == 0) or self.state.binomial(1, self.random_infection_chance):
            self._infect_random()

        new_infections = self._infect_neighbours()
        deaths, recoveries = self._conclude_all()
        self.observation_space.test_population(self._step)
        known_new_infections = self.observation_space.update_observed_statuses(self._step)
        self._update_immunities()

        score = self.scoring.score(graph=self.observation_space.graph,
                                   new_infections=new_infections,
                                   new_deaths=deaths)
        obs_score = self.scoring.score(graph=self.observation_space,
                                       new_infections=known_new_infections,
                                       new_deaths=deaths)

        self._log(new_infections=known_new_infections,
                  known_new_infections=known_new_infections,
                  deaths=deaths,
                  recoveries=recoveries,
                  score=score,
                  obs_score=obs_score)

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
        self._total_steps += steps
        t0 = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=(UserWarning, RuntimeWarning))
            for _ in tqdm(range(steps), desc=self.name):
                self.step()

                if plot or save:
                    self.plot(show=plot,
                              save=save)

        print(f"Ran {steps} steps in {np.round(time.time() - t0, 2)}s")

    def clone(self) -> "Population":
        """Clone a fresh object with same seed (could be None)."""
        return Population(disease=self.disease.clone(),
                          observation_space=self.observation_space.clone(),
                          healthcare=self.healthcare.clone(),
                          scoring=self.scoring.clone(),
                          name=self.name,
                          seed=self.seed,
                          random_infection_chance=self.random_infection_chance,
                          plot_both=self.plot_both,
                          plot_ts_fields_g1=self.plot_ts_fields_g1,
                          plot_ts_fields_g2=self.plot_ts_fields_g2,
                          plot_ts_obs_fields_g1=self.plot_ts_obs_fields_g1,
                          plot_ts_obs_fields_g2=self.plot_ts_obs_fields_g2)
