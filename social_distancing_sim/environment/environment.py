import time
import warnings
from dataclasses import dataclass
from typing import List, Tuple, Union, Any, Dict

import numpy as np
from tqdm import tqdm

from social_distancing_sim.environment.disease import Disease
from social_distancing_sim.environment.action_space import ActionSpace
from social_distancing_sim.environment.environment_plotting import EnvironmentPlotting
from social_distancing_sim.environment.healthcare import Healthcare
from social_distancing_sim.environment.history import History
from social_distancing_sim.environment.observation_space import ObservationSpace
from social_distancing_sim.environment.scoring import Scoring


@dataclass
class Environment:
    observation_space: ObservationSpace
    action_space: ActionSpace = ActionSpace()
    disease: Disease = Disease()
    healthcare: Healthcare = Healthcare()
    scoring: Scoring = Scoring()
    environment_plotting: EnvironmentPlotting = None  # This is mutable, and will be changed don't init for default!!
    name: str = "unnamed_population"
    seed: Union[None, int] = None

    initial_infections: int = 2
    random_infection_chance: float = 0.01

    def __post_init__(self) -> None:
        self._prepare_random_state()

        self.total_population = self.observation_space.graph.total_population
        self._step: int = 0
        self.history: History[str, List[int]] = History.with_defaults()
        self._total_steps: int = 0

        if self.environment_plotting is None:
            self.environment_plotting = EnvironmentPlotting()
        self.environment_plotting.prepare_output_path(name=self.name)

    def _prepare_random_state(self) -> None:
        self._random_state = np.random.RandomState(seed=self.seed)

    def _infect_random(self) -> None:
        """Infect a random node."""
        node_id = self.observation_space.graph.current_clear_nodes[
            self._random_state.randint(0, len(self.observation_space.graph.current_clear_nodes))]
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

            # Outcome is either recovery, death, or continuationddd
            if node["alive"]:
                if node["infected"] == 0:
                    recoveries += 1
            else:
                deaths += 1

        return deaths, recoveries

    def _log(self, new_infections: int, known_new_infections: int, deaths: int, recoveries: int,
             turn_score: float = 0.0,
             obs_turn_score: float = 0.0) -> None:

        # Log counts/score for this turn
        self.history.log({"Turn score": turn_score,
                          "Observed turn score": obs_turn_score,
                          "New infections": new_infections,
                          "Known new infections": known_new_infections,
                          "New deaths": deaths,
                          "Current recovered": recoveries})

        # Log full space and observed space
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
                          "Total recovered": np.sum(self.history["Current recoveries"]),
                          "Total infections": np.sum(self.history["New infections"]),
                          "Known total infections": np.sum(self.history["Known new infections"]),
                          "Overall score": np.sum(self.history["Turn score"]),
                          "Observed overall score": np.sum(self.history["Observed turn score"])})

        # Dependent
        self.history.log({"Current infection prop": self.history["Current infections"][-1] / self.total_population,
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

    def _update_immunities(self):
        for node in self.observation_space.graph.current_immune_nodes:
            self.disease.decay_immunity(self.observation_space.graph.g_.nodes[node])

    def _select_random_nodes(self, n: int) -> int:
        return self._random_state.choice(self.observation_space.graph.g_.nodes, size=n, replace=True)

    def _act(self, actions: Dict[int, str]) -> Tuple[Dict[int, str], float]:

        completed_actions = {}
        total_action_cost = 0
        for target_node_id, ac in actions.items():
            total_action_cost = getattr(self.action_space, ac)(target_node_id=target_node_id,
                                                               env=self, step=self._step)
            completed_actions.update({target_node_id: ac})

        return completed_actions, total_action_cost

    def step(self, actions: Dict[int, str]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        self.observation_space.reset_cached_values()
        self.observation_space.graph.reset_cached_values()

        done = False

        # Run some env
        # Initial infections
        if self._step == 0:
            for _ in range(self.initial_infections):
                self._infect_random()
        # Random infections
        if self._random_state.binomial(1, self.random_infection_chance):
            self._infect_random()

        # Act
        completed_actions, action_costs = self._act(actions)

        # Run remaining env
        new_infections = self._infect_neighbours()
        deaths, recoveries = self._conclude_all()
        self.observation_space.test_population(self._step)
        known_new_infections = self.observation_space.update_observed_statuses(self._step)
        self._update_immunities()

        # Score and log complete env history
        turn_score = self.scoring.score_turn(graph=self.observation_space.graph,
                                             action_cost=action_costs,
                                             new_infections=new_infections,
                                             new_deaths=deaths)
        obs_turn_score = self.scoring.score_turn(graph=self.observation_space,
                                                 action_cost=action_costs,
                                                 new_infections=known_new_infections,
                                                 new_deaths=deaths)

        self._log(new_infections=new_infections,
                  known_new_infections=known_new_infections,
                  deaths=deaths,
                  recoveries=recoveries,
                  turn_score=turn_score,
                  obs_turn_score=obs_turn_score)

        self._step += 1

        # Gym api
        observation = {'obs': self.observation_space,
                       'turn_score': turn_score,
                       'obs_score': obs_turn_score,
                       'action_costs': action_costs,
                       'completed_actions': completed_actions}
        reward = obs_turn_score
        if self._step == self._total_steps:
            done = True
        info = {}

        return observation, reward, done, info

    def run(self, steps: 10,
            plot: bool = True,
            save: bool = True) -> None:
        """
        Run a passive simulation for a number of iterations

        :param steps: Number of steps to run.
        :param plot: Display plots while running.
        :param save: Save plot of each step while running.
        """
        self._total_steps += steps
        t0 = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=(UserWarning, RuntimeWarning))
            for _ in tqdm(range(steps), desc=self.name):
                self.step(actions={})

                if plot or save:
                    self.environment_plotting.plot(obs=self.observation_space,
                                                   history=self.history,
                                                   healthcare=self.healthcare,
                                                   step=self._step,
                                                   total_steps=self._total_steps,
                                                   show=plot,
                                                   save=save)

        print(f"Ran {steps} steps in {np.round(time.time() - t0, 2)}s")

    def replay(self, duration: float = 0.1):
        self.environment_plotting.replay(duration=duration)

    def clone(self) -> "Environment":
        """Clone a fresh object with same seed (could be None)."""
        return Environment(disease=self.disease.clone(),
                           observation_space=self.observation_space.clone(),
                           healthcare=self.healthcare.clone(),
                           scoring=self.scoring.clone(),
                           environment_plotting=self.environment_plotting.clone(),
                           name=self.name,
                           seed=self.seed)

    @property
    def state(self) -> np.ndarray:
        return np.concatenate([nv['status'].state for _, nv in self.observation_space.graph.g_.nodes.data()])
