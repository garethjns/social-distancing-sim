import copy
import pickle
from typing import Union, List, Tuple, Dict

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

from social_distancing_sim.agent.agent_base import AgentBase
from social_distancing_sim.gym.agent.rl.epsilon import Epsilon
from social_distancing_sim.gym.agent.rl.replay_buffer import ReplayBuffer
from social_distancing_sim.gym.gym_env import GymEnv


class DeepQAgent(AgentBase):
    def __init__(self, env: GymEnv,
                 epsilon: Epsilon = None,
                 replay_buffer: ReplayBuffer = None,
                 gamma: float = 0.98,
                 model_unit_scale: float = 1,
                 replay_buffer_samples=75,
                 *args, **kwargs) -> None:

        super().__init__(*args, **kwargs)
        self.env = env
        self.gamma = gamma
        if replay_buffer is None:
            replay_buffer = ReplayBuffer()
        self.replay_buffer = replay_buffer
        if epsilon is None:
            epsilon = Epsilon()
        self.epsilon = epsilon
        self.model_unit_scale = model_unit_scale
        self.replay_buffer_samples = replay_buffer_samples
        self._prep_pp()
        self._prep_models()

    def _prep_models(self):
        self._policy_model = self._build_model('policy_model')
        self._target_model = self._build_model('target_model')

    def _prep_pp(self) -> None:
        # Sample observations from env and for pipeline
        obs = np.array([self.env.observation_space[0].sample() for _ in range(1000)])

        pipe = Pipeline([('ss', StandardScaler())])
        pipe.fit(obs)

        self.pp = pipe

    def _build_model(self, model_name: str) -> keras.Model:

        conv_shape = self.env.observation_space[1].sample().shape

        fc_input = keras.layers.Input(name='fc_input', shape=self.env.observation_space[0].shape)
        fc1 = keras.layers.Dense(units=12, name='fc1', activation='relu')(fc_input)

        conv_input = keras.layers.Input(name='conv_input', shape=(conv_shape[0], conv_shape[1], 1))
        conv1 = keras.layers.Conv2D(24, kernel_size=(6, 6),
                                    name='conv1', activation='relu', dtype=np.float32)(conv_input)
        conv2 = keras.layers.Conv2D(12, kernel_size=(3, 3), name='conv2', activation='relu')(conv1)
        flatten = keras.layers.Flatten(name='flatten')(conv2)
        concat = keras.layers.Concatenate(name='concat')([fc1, flatten])

        fc2 = keras.layers.Dense(units=64, name='fc2', activation='relu')(concat)
        fc3 = keras.layers.Dense(units=16, name='fc3', activation='relu')(fc2)
        output = keras.layers.Dense(units=self.env.action_space.n, name='output', activation=None)(fc3)

        opt = keras.optimizers.Adam(learning_rate=0.001)
        model = keras.Model(inputs=[fc_input, conv_input], outputs=[output],
                            name=model_name)
        model.compile(opt, loss='mse')

        return model

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
            state = state[0:2]

        return state

    def transform(self, s: Tuple[np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        s = self._filter_state(s)

        s_ = s[1]
        if len(s_.shape) < 3:
            s_ = np.expand_dims(s_, axis=0)
        if len(s_.shape) < 4:
            s_ = np.expand_dims(s_, axis=3)

        return self.pp.transform(np.atleast_2d(s[0])), s_.astype(np.float32)

    def update(self, state1: Tuple[np.ndarray, np.ndarray], action: int, reward: float, done: bool,
               state2: Tuple[np.ndarray, np.ndarray]) -> None:
        """
        Update agent: s -> a -> r, s_.
        """

        # Add sars to experience buffer
        self.replay_buffer.append((state1, action, reward, done))

        # If buffer is too small, don't train
        if len(self.replay_buffer) < self.replay_buffer.replay_buffer_size:
            return

        ss, aa, rr, dd, ss_ = self.replay_buffer.sample(self.replay_buffer_samples)

        states1 = (np.stack([s[0] for s in ss]), np.stack([s[1] for s in ss]))
        states2 = (np.stack([s[0] for s in ss_]), np.stack([s[1] for s in ss_]))
        y_now = self._target_model.predict(self.transform(states1))
        y_future = self._target_model.predict(self.transform(states2))
        x = states1
        y = []
        for i, (state, action, reward, done, state_) in enumerate(zip(ss, aa, rr, dd, ss_)):
            if done:
                g = reward
            else:
                g = reward + self.gamma * np.max(y_future[i, :])

            # Set non-acted actions to y_now preds and acted action to y_future pred
            y_ = y_now[i, :]
            # NB: No index here are wel just killed that dimension
            y_[action] = g
            y.append(y_)

        # Fit main model
        x = (np.stack(x[0]), np.stack(x[1]))
        y = np.stack(y)
        self._policy_model.fit(self.transform(x), y,
                               batch_size=self.replay_buffer_samples,
                               verbose=0)

    def set_env(self, *args, **kwargs):
        """Pass for compatibility with set env used in AgentBase. Not necessary here as this agent only uses the env
        to sample examples."""
        pass

    def _select_actions_targets(self) -> Dict[int, int]:
        """This agent only selects actions, not targets."""
        pass

    def get_best_action(self, s) -> np.ndarray:
        preds = self._policy_model.predict(self.transform(s))

        return np.argmax(preds)

    def get_actions(self,
                    state: Union[Tuple[np.ndarray, np.ndarray],
                                 Tuple[np.ndarray, np.ndarray, np.ndarray]],
                    training: bool = False) -> Tuple[List[int], None]:
        """Get actions for current state."""

        state = self._filter_state(state)
        action = self.epsilon.select(greedy_option=lambda: self.get_best_action(state),
                                     random_option=lambda: self.env.action_space.sample(),
                                     training=training)

        # This agent doesn't set targets
        return [action] * self.actions_per_turn, None

    def update_action_model(self):
        self._target_model.set_weights(self._policy_model.get_weights())

    def save(self, fn: str):
        model_to_save = copy.deepcopy(self)
        model_to_save._policy_model = None
        model_to_save._target_model = None

        name = fn.split('.')[0]
        self._policy_model.save(f"{name}.h5")
        pickle.dump(model_to_save, open(f"{name}.pkl", "wb"))

    @classmethod
    def load(cls, fn: str) -> "DeepQAgent":
        name = fn.split('.')[0]
        loaded_model = pickle.load(open(f"{name}.pkl"))
        keras.models.load_model(f"{name}.h5")

        return loaded_model

    def clone(self) -> "DeepQAgent":
        return copy.deepcopy(self)
