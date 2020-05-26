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
                 replay_buffer_samples=75,
                 dueling: bool = True,
                 *args, **kwargs) -> None:

        super().__init__(*args, **kwargs)
        self.env = env
        self.gamma = gamma
        self.dueling = dueling
        if replay_buffer is None:
            replay_buffer = ReplayBuffer()
        self.replay_buffer = replay_buffer
        if epsilon is None:
            epsilon = Epsilon()
        self.epsilon = epsilon
        self.replay_buffer_samples = replay_buffer_samples
        self._fn_used_to_load: str = ''
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

        graph_shape = self.env.observation_space[1].sample().shape
        graph_nodes = graph_shape[0] * graph_shape[1]

        summary_input = keras.layers.Input(name='summary_input', shape=self.env.observation_space[0].shape)
        summary_fc1 = keras.layers.Dense(units=12, name='summary_fc1', activation='relu')(summary_input)

        graph_input = keras.layers.Input(name='conv_input', shape=(graph_shape[0], graph_shape[1], 1))
        flatten = keras.layers.Flatten(name='flatten')(graph_input)
        graph_fc1 = keras.layers.Dense(units=int(graph_nodes), name='graph_fc1', activation='relu')(flatten)
        graph_fc2 = keras.layers.Dense(units=int(graph_nodes / 2), name='graph_fc2', activation='relu')(graph_fc1)
        graph_fc3 = keras.layers.Dense(units=int(graph_nodes / 4), name='graph_fc3', activation='relu')(graph_fc2)

        concat = keras.layers.Concatenate(name='concat')([summary_fc1, graph_fc3])
        fc1 = keras.layers.Dense(units=64, name='fc2', activation='relu')(concat)
        fc2 = keras.layers.Dense(units=16, name='fc3', activation='relu')(fc1)

        if self.dueling:
            # Using dueling architecture (split value and action advantages)
            v_layer = keras.layers.Dense(1, activation='linear')(fc2)
            a_layer = keras.layers.Dense(self.env.action_space.n, activation='linear')(fc2)

            def merge_layer(layer_inputs):
                return layer_inputs[0] + layer_inputs[1] - keras.backend.mean(layer_inputs[1], axis=1, keepdims=True)

            output = keras.layers.Lambda(merge_layer, output_shape=(self.env.action_space.n,),
                                         name="output")([v_layer, a_layer])
        else:
            output = keras.layers.Dense(units=self.env.action_space.n, name='output', activation=None)(fc2)

        opt = keras.optimizers.Adam(learning_rate=0.001)
        model = keras.Model(inputs=[summary_input, graph_input], outputs=[output],
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
        """Pass for compatibility with set env used in AgentBase. Not necessary here"""
        pass

    def _select_actions_targets(self) -> Dict[int, int]:
        """This agent only selects actions, not targets."""
        pass

    def get_best_action(self, s) -> np.ndarray:
        preds = self._policy_model.predict(self.transform(s))

        return np.argmax(preds)

    def get_actions(self,
                    state: Union[Tuple[np.ndarray, np.ndarray],
                                 Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
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
        self._policy_model.save(f"{fn}.h5")
        self._policy_model = None
        self._target_model = None

        agent_to_save = copy.deepcopy(self)
        pickle.dump(agent_to_save, open(f"{fn}.pkl", "wb"))

        self._policy_model = keras.models.load_model(f"{fn}.h5")
        self._target_model = keras.models.load_model(f"{fn}.h5")

    @classmethod
    def load(cls, fn: str) -> "DeepQAgent":
        agent = pickle.load(open(f"{fn}.pkl", 'rb'))
        model = keras.models.load_model(f"{fn}.h5")

        agent._policy_model = model
        agent._target_model = model

        agent._fn_used_to_load = fn
        return agent

    def clone(self) -> "DeepQAgent":
        # TODO: Can't copy models, bodge for now. This should work for multi sims where it's used, but obviously isn't
        #       the best solution....
        clone = DeepQAgent.load(fn=self._fn_used_to_load)
        return clone
