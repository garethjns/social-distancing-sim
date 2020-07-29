from dataclasses import dataclass
from typing import Iterable, Dict, Any, Callable, Union

import gym
from reinforcement_learning_keras.agents.components.history.training_history import TrainingHistory
from reinforcement_learning_keras.agents.components.replay_buffers.continuous_buffer import ContinuousBuffer
from reinforcement_learning_keras.agents.q_learning.exploration.epsilon_greedy import EpsilonGreedy

from social_distancing_sim.agent.rl_agents.models.untargeted_nn import UntargetedNN


@dataclass
class RLKAgentConfigs:
    """Default configs to save some typing."""
    env_spec: str
    expected_obs_shape: Iterable[int]
    n_actions: int
    agent_name: str = 'unnamed_agent'
    plot_during_training: bool = True
    env_wrappers: Iterable[Union[Callable, gym.Wrapper]] = ()

    @property
    def _default_training_history_kwargs(self) -> Dict[str, Any]:
        return {"plotting_on": self.plot_during_training,
                "plot_every": 25, "rolling_average": 12,
                'agent_name': self.agent_name}

    def build_for_dqn_untargeted(self):
        return {'name': self.agent_name,
                'env_spec': self.env_spec,
                'training_history': TrainingHistory(**self._default_training_history_kwargs),
                'model_architecture': UntargetedNN(observation_shape=self.expected_obs_shape,
                                                   n_actions=self.n_actions, output_activation=None,
                                                   opt='adam', learning_rate=0.001),
                'env_wrappers': self.env_wrappers,
                'gamma': 0.99,
                'final_reward': -200,
                'replay_buffer_samples': 32,
                'eps': EpsilonGreedy(eps_initial=0.3),
                'replay_buffer': ContinuousBuffer(buffer_size=5000)}

    def build_for_dqn_targeted(self):
        raise NotImplementedError

    def build_for_linear_untargeted(self):
        return {'name': self.agent_name,
                'env_spec': self.env_spec,
                'training_history': TrainingHistory(**self._default_training_history_kwargs),
                'env_wrappers': self.env_wrappers,
                'gamma': 0.99,
                'log_exemplar_space': False,
                'final_reward': -200,
                'eps': EpsilonGreedy(eps_initial=0.4, eps_min=0.01)}

    def build_for_linear_targeted(self):
        raise NotImplementedError
