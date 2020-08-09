import logging
import os
import pprint
import shutil
import time
import warnings
from dataclasses import dataclass
from typing import List, Tuple, Union, Any, Dict

import numpy as np
from tqdm import tqdm

from social_distancing_sim.environment.action_space import ActionSpace
from social_distancing_sim.environment.disease import Disease
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
        # Output path is only used for log file, so it's prepared along with the logger
        # (Plot path is handled in EnvironmentPlotting)
        self.output_path: Union[None, str] = None
        self._prepare_logger(to_file=False)

        self._prepare_random_state()

        self.total_population = self.observation_space.graph.total_population
        self._step: int = 0
        self.history: History[str, List[int]] = History.with_defaults()
        self._total_steps: int = 0

        self.set_output_path()
        if self.environment_plotting is None:
            self.environment_plotting = EnvironmentPlotting(name=self.output_path)

    def set_output_path(self, path: str = None) -> None:
        """
        Set the path used for any output.

        This only includes log file, so only actually created if logger is used. Or if EnvironmentPlotting decides to
        create the graph path in the same dir.
        """
        if path is None:
            path = self.name
        path = f"{os.path.abspath(path)}".replace('\\', '/')
        self.output_path = path

        # Set plotting output if it hasn't been specifically set already
        if (self.environment_plotting is not None) and (self.environment_plotting.name is None):
            self.environment_plotting.name = self.name

    def _prepare_logger(self, to_file: bool = False) -> None:
        logger = logging.getLogger(self.name)
        logger.setLevel(logging.INFO)
        self.logger = logger
        self.log_to_file = to_file

    @property
    def log_to_file(self) -> bool:
        return self._log_to_file

    @log_to_file.setter
    def log_to_file(self, on: bool):
        if on and (not self._log_to_file):
            # Create a new handler and new log file. Create output dir if it doesn't exist.
            # If already on, continue with previous log file (ie. doesn't enter here).
            self.set_output_path(self.output_path)
            os.makedirs(self.output_path, exist_ok=True)
            self.log_file = os.path.join(self.output_path, 'log.txt').replace('\\', '/')

            handler = logging.FileHandler(self.log_file, 'w')
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            handler.setLevel(logging.INFO)

            self.logger.handlers = []
            self.logger.addHandler(handler)
            self._log_to_file = True
        else:
            self.logger.handlers = []
            self._log_to_file = False

    def _prepare_random_state(self) -> None:
        self._random_state = np.random.RandomState(seed=self.seed)

    def _infect_random(self) -> None:
        """Infect a random node, if possible."""
        if len(self.observation_space.graph.current_clear_nodes) > 0:
            node_id = self.observation_space.graph.current_clear_nodes[
                self._random_state.randint(0, len(self.observation_space.graph.current_clear_nodes))]
            self.disease.force_infect(self.observation_space.graph.g_.nodes[node_id])

            self.logger.info(f"Randomly infected node: {node_id}")

    def _infect_neighbours(self) -> int:
        """For all the currently infected nodes, attempt to infect neighbours."""
        total_new_infections = 0
        for n in self.observation_space.graph.current_infected_nodes:
            neighbours = list(self.observation_space.graph.g_.neighbors(n))

            new_infections = self.disease.try_to_infect_multiple(
                source_node=self.observation_space.graph.g_.nodes.data()[n],
                target_nodes=[self.observation_space.graph.g_.nodes.data()[neighbour]
                              for neighbour in neighbours])
            self.logger.info(
                f"Node {n} infected nodes {[node for node, new in zip(neighbours, new_infections) if new]}")

            total_new_infections += sum(new_infections)

        return total_new_infections

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
            outcome = 'continues'
            node = self.disease.conclude(self.observation_space.graph.g_.nodes[n],
                                         recovery_rate_modifier=recovery_rate_modifier)

            # Outcome is either recovery, death, or continuation
            if node["alive"]:
                if node["infected"] == 0:
                    recoveries += 1
                    outcome = "recovered"
            else:
                deaths += 1
                outcome = "died"

            self.logger.info(f"Node {n} disease outcome: {outcome.capitalize()}")

        return deaths, recoveries

    def _update_immunities(self):
        for node in self.observation_space.graph.current_immune_nodes:
            current_immunity = self.observation_space.graph.g_.nodes[node]['immune']
            self.disease.decay_immunity(self.observation_space.graph.g_.nodes[node])
            new_immunity = self.observation_space.graph.g_.nodes[node]["immune"]
            self.logger.info(f"Decayed immunity for node {node}: {current_immunity} -> {new_immunity}")

    def _select_random_nodes(self, n: int) -> int:
        return self._random_state.choice(self.observation_space.graph.g_.nodes, size=n, replace=True)

    def select_reasonable_targets(self, actions: List[int]) -> Dict[int, int]:
        """Select random, but appropriate target node for a list of actions."""

        suggested_targets = {
            0: [],  # Nothing
            1: self.observation_space.current_clear_nodes,  # Vaccinate
            2: list(  # Isolate
                set(self.observation_space.current_infected_nodes).difference(
                    self.observation_space.current_isolated_nodes)),
            3: list(  # Reconnect
                set(self.observation_space.current_clear_nodes).intersection(
                    self.observation_space.current_isolated_nodes)),
            4: self.observation_space.current_infected_nodes,  # Treat
            5: self.observation_space.current_alive_nodes,  # Provide mask
            6: self.observation_space.current_masked_nodes}  # Remove mask

        targets = []
        acts, counts = np.unique(actions,
                                 return_counts=True)
        for act, count in zip(acts, counts):
            targets.extend(self.action_space.select_random_target(n=count, available_targets=suggested_targets[act]))

        if len(targets) != len(actions):
            raise ValueError

        actions_dict = {t: a for t, a in zip(targets, actions)}
        self.logger.info(f"Environment assigned actions to targets automatically: {actions_dict}")

        # Remove actions with invalid targets
        return {t: a for t, a in actions_dict.items() if t != -1}

    def _act(self, actions: List[int], targets: List[int] = None) -> Tuple[Dict[int, int], float]:
        # If no targets supplied, select automatically
        if targets is None:
            targets = []
        if len(targets) == 0:
            actions_dict = self.select_reasonable_targets(actions)
        else:
            actions_dict = {t: a for t, a in zip(targets, actions)}

        self.logger.info(f"Actions dict for turn: {actions_dict}")

        # Perform actions
        completed_actions = {}
        total_action_cost = 0
        for target_node_id, ac in actions_dict.items():
            ac_name = self.action_space.get_action_name(ac)
            action_cost = getattr(self.action_space, ac_name)(target_node_id=target_node_id,
                                                              env=self, step=self._step)
            action_taken = {target_node_id: ac}

            completed_actions.update(action_taken)
            total_action_cost += action_cost
            self.logger.info(f"Action taken: {action_taken}, costing {action_cost}")

        return completed_actions, total_action_cost

    def step(self, actions: List[int], targets: [List[Union[None, int]]] = None) -> Tuple[Dict[str, Any], float, bool]:
        """
        Run step with optional actions (and targets).

        :param actions:
        :param targets:
        """

        self.logger.info(f"\n\n***Step: {self._step}***")
        self.observation_space.reset_cached_values()
        self.observation_space.graph.reset_cached_values()
        done = False

        # Run some env
        # Initial infections
        if self._step == 0:
            self.logger.info(f"Applying {self.initial_infections} initial infections...")
            for _ in range(self.initial_infections):
                self._infect_random()
        # Random infections
        if self._random_state.binomial(1, self.random_infection_chance):
            self.logger.info(f"Applying random infections...")
            self._infect_random()

        # Act
        self.logger.info(f"Requested actions: {actions}, with targets {targets}")
        completed_actions, action_costs = self._act(actions, targets)
        self.logger.info(f"Action summary: Completed actions: {completed_actions}, total cost: {action_costs}")

        # Run remaining env
        new_infections = self._infect_neighbours()
        self.logger.info(f"Infection summary: New infections: {new_infections}")
        deaths, recoveries = self._conclude_all()
        self.logger.info(f"Disease conclusion summary: Deaths: {deaths}, Recoveries: {recoveries}")
        self.observation_space.test_population(self._step)
        known_new_infections = self.observation_space.update_observed_statuses(self._step)
        self.logger.info(f"Infections found in testing: {known_new_infections}")
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
        self.logger.info(f"Turn score: {np.round(turn_score, 2)}")
        self.logger.info(f"Observed turn score: {np.round(obs_turn_score, 2)}")

        self.history.log_score(new_infections=new_infections,
                               known_new_infections=known_new_infections, deaths=deaths,
                               recoveries=recoveries, turn_score=turn_score,
                               obs_turn_score=obs_turn_score)
        self.history.log_actions(actions_taken=completed_actions,
                                 actions_attempted={a: None for a in actions})
        self.history.log_observation_space(obs=self.observation_space, healthcare=self.healthcare)
        self.logger.info(f"Turn summary:\n{pprint.pformat(self.history.last_turn)}")

        self._step += 1

        observation = {'obs': self.observation_space,
                       'history': self.history,
                       'healthcare': self.healthcare,
                       'turn_score': turn_score,
                       'obs_score': obs_turn_score,
                       'action_costs': action_costs,
                       'completed_actions': completed_actions}
        if self._step == self._total_steps:
            done = True
            self.logger.info(f"Done = {done}")

        return observation, obs_turn_score, done

    def plot(self, plot: bool = True, save: bool = False):
        if plot or save:
            self.environment_plotting.plot(obs=self.observation_space,
                                           history=self.history,
                                           healthcare=self.healthcare,
                                           step=self._step,
                                           total_steps=self._total_steps,
                                           show=plot, save=save)

    def run(self, steps: int,
            plot: bool = True,
            save: bool = True) -> None:
        """
        Run a passive simulation for a number of iterations.

        :param steps: Number of steps to run.
        :param plot: Display plots while running.
        :param save: Save plot of each step while running.
        """
        self.logger.info(f"Running passive simulation with {steps}=steps, plot={plot}, save={save}")
        self.plot(plot=plot, save=save)
        self._total_steps += steps
        t0 = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            warnings.simplefilter("ignore", category=RuntimeWarning)
            for _ in tqdm(range(steps), desc=self.name):
                self.step(actions=[])
                self.plot(plot=plot, save=save)

        print(f"Ran {steps} steps in {np.round(time.time() - t0, 2)}s")

    def replay(self, duration: float = 0.1):
        self.environment_plotting.replay(duration=duration)

    def clone(self) -> "Environment":
        """Clone a fresh object with same seed (could be None)."""
        return Environment(disease=self.disease.clone(),
                           action_space=self.action_space.clone(),
                           observation_space=self.observation_space.clone(),
                           healthcare=self.healthcare.clone(),
                           scoring=self.scoring.clone(),
                           environment_plotting=self.environment_plotting.clone(),
                           name=self.name,
                           seed=self.seed,
                           initial_infections=self.initial_infections,
                           random_infection_chance=self.random_infection_chance)

    @property
    def state(self) -> np.ndarray:
        return np.concatenate([nv['status'].state for _, nv in self.observation_space.graph.g_.nodes.data()])
