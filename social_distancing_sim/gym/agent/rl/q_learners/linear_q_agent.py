import copy
import pickle
from typing import Dict, List, Tuple, Union

import numpy as np
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler

from social_distancing_sim.agent.agent_base import AgentBase
from social_distancing_sim.gym.agent.rl.epsilon import Epsilon
from social_distancing_sim.gym.gym_env import GymEnv


class LinearQAgent(AgentBase):
    """
    This agent assumes all actions are available at all times.

    The selected action is duplicated actions_per_turn times with no selected target.

    # TODO: This doesn't use, but is compatible with the sds.AgentBase. Some functionality can probably be combined.
    # TODO: Even if not merged with AgentBase, will definitely share functionality with other RL agents, should define
            interface.
    """

    def __init__(self, env: GymEnv,
                 gamma: float = 0.98,
                 rb_components: List[Tuple[float, int]] = None,
                 epsilon: Epsilon = None,
                 *args, **kwargs) -> None:

        super().__init__(*args, **kwargs)

        self.env = env
        self.gamma = gamma
        if epsilon is None:
            epsilon = Epsilon()
        self.epsilon = epsilon
        if rb_components is None:
            self.use_rbf = True
            self.rb_components = [(100, 6), (1, 6), (0.02, 6)]
        else:
            self.use_rbf = False
            self.rb_components = None
        self._prep_pp()
        self._prep_mods()

    def set_env(self, *args, **kwargs):
        """Pass for compatibility with set env used in AgentBase. Not necessary here as this agent only uses the env
        to sample examples."""
        pass

    def _prep_pp(self) -> None:
        if self.use_rbf:
            # Prep pipeline with scaler and rbfs paths
            fu = FeatureUnion([(f'rbf{i}', RBFSampler(gamma=g, n_components=n))
                               for i, (g, n) in enumerate(self.rb_components)])
            pipe = Pipeline([('ss', StandardScaler()),
                             ('rbfs', fu)])
        else:
            pipe = StandardScaler()

        # Sample observations from env and fit pipeline
        obs = np.array([self.env.observation_space[0].sample() for _ in range(1000)])
        pipe.fit(obs)

        self.pp = pipe

    def _prep_mods(self):
        mods = {a: SGDRegressor() for a in range(self.env.action_space.n)}
        for mod in mods.values():
            mod.partial_fit(self.transform(self.env.reset()), [0])

        self.mods = mods

    def transform(self, s: np.ndarray) -> np.ndarray:
        s = self._filter_state(s)
        return self.pp.transform(np.atleast_2d(s))

    def update(self, state1: np.ndarray, action: int, reward: float, done: bool, state2: np.ndarray):
        """Update model with single observation."""
        if done:
            g = reward
        else:
            g = reward + self.gamma * np.max(list(self.predict(state2).values()))

        # Actions are all the same with this agent
        self.partial_fit(state1, action, g)

    def partial_fit(self, s, a, g) -> None:
        x = self.transform(s)
        self.mods[a].partial_fit(x, [g])

    def predict(self, s) -> Dict[int, float]:
        s = self.transform(s)

        return {a: float(mod.predict(s)) for a, mod in self.mods.items()}

    @staticmethod
    def _filter_state(state) -> np.ndarray:
        """
        This agent is designed to train in a SummaryObservationWrapper where state is passed as a single (6,) np.array,
        but might be run in a Sim using a sds.environment.Environment, which sends the whole state observation space
        (currently Tuple(3) with state[0] containing the expected (6,) array).

        Static for now, but it may be worth generalising for future wrapper chains.

        :return: The filtered state, as per wrapper chain (specifically SummaryObservationWrapper for now).
        """

        # If the state isn't an array, assume it's the full observation space and get the summary part used by this
        # agent.
        if not isinstance(state, np.ndarray):
            state = state[0]

        return state

    def get_actions(self, state: Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
                    training: bool = False) -> Tuple[List[int], None]:
        """Get actions for current state."""

        state = self._filter_state(state)
        action = self.epsilon.select(greedy_option=lambda: self.get_best_action(state),
                                     random_option=lambda: self.env.action_space.sample(),
                                     training=training)

        # This agent doesn't set targets
        return [action] * self.actions_per_turn, None

    def get_best_action(self, s: np.ndarray) -> int:
        preds = self.predict(s)

        best_action_value = -np.inf
        best_action = 0
        for k, v in preds.items():
            if v > best_action_value:
                best_action_value = v
                best_action = k

        return best_action

    def _select_actions_targets(self) -> Dict[int, int]:
        """This agent only selects actions, not targets."""
        pass

    def save(self, fn: str):
        pickle.dump(self, open(fn, 'wb'))

    @classmethod
    def load(cls, fn: str) -> "LinearQAgent":
        return pickle.load(open(fn, 'rb'))

    def clone(self) -> "LinearQAgent":
        return copy.deepcopy(self)
