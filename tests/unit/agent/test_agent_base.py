import unittest
from unittest.mock import patch, MagicMock

from social_distancing_sim.agent.non_learning_agent_base import NonLearningAgentBase
from social_distancing_sim.environment.action_space import ActionSpace
from social_distancing_sim.environment.environment import Environment
from social_distancing_sim.environment.gym.gym_env import GymEnv
from social_distancing_sim.environment.observation_space import ObservationSpace


class TestAgentBase(unittest.TestCase):

    def setUp(self):
        mock_obs = unittest.mock.MagicMock(spec=ObservationSpace)
        mock_obs.current_alive_nodes = [1, 2, 3]
        mock_action = MagicMock(spec=ActionSpace)
        mock_action.available_action_ids = [1, 2]

        mock_env = MagicMock(spec=GymEnv)
        mock_env.sds_env = MagicMock(spec=Environment)
        mock_env.sds_env.observation_space = mock_obs
        mock_env.sds_env.action_space = mock_action

        self.mock_env = mock_env

    @patch.multiple(NonLearningAgentBase, __abstractmethods__=set())
    def test_sample_action(self):
        # Arrange
        agent = NonLearningAgentBase(self.mock_env,
                                     actions_per_turn=1)

        # Act
        actions = agent.sample()

        # Assert
        self.assertIsInstance(actions, dict)
        self.assertEqual(1, len(actions.keys()))

    @patch.multiple(NonLearningAgentBase, __abstractmethods__=set())
    def test_limit_actions_by_available_targets(self):
        # Arrange
        agent = NonLearningAgentBase(self.mock_env,
                                     actions_per_turn=6)

        # Act
        actions = agent.sample()

        # Assert
        self.assertIsInstance(actions, dict)
        self.assertEqual(3, len(actions.keys()))
