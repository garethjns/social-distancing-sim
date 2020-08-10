import copy
import sys
import unittest
from functools import partial
from typing import Union

import gym
import numpy as np

from social_distancing_sim.agent import DistancingPolicyAgent, DummyAgent, MultiAgent, VaccinationPolicyAgent, \
    TreatmentPolicyAgent, MaskingPolicyAgent
from social_distancing_sim.agent.basic_agents.vaccination_agent import VaccinationAgent
from social_distancing_sim.agent.learning_agent_base import LearningAgentBase
from social_distancing_sim.agent.non_learning_agent_base import NonLearningAgentBase
from social_distancing_sim.agent.rl_agents.q_learning.dqn_untargeted import DQNUntargeted
from social_distancing_sim.agent.rl_agents.rlk_agent_configs import RLKAgentConfigs
from social_distancing_sim.environment.gym.gym_env import GymEnv
from social_distancing_sim.environment.gym.wrappers.flatten_obs_wrapper import FlattenObsWrapper
from social_distancing_sim.environment.gym.wrappers.limit_obs_wrapper import LimitObsWrapper
from tests.common.env_fixtures import register_test_envs


class TestGymEnv(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        register_test_envs()

    @staticmethod
    def _passive_run_for(env: GymEnv, steps: int = 25) -> GymEnv:
        _ = env.reset()
        for _ in range(steps):
            env.step(([], []))

        return env

    @staticmethod
    def _active_run_for(agent: Union[NonLearningAgentBase, LearningAgentBase],
                        steps: int = 25) -> Union[NonLearningAgentBase, LearningAgentBase]:
        """This loop defines Sim runs, all agents should work with this."""
        state = agent.env.reset()
        for _ in range(steps):
            actions, targets = agent.get_actions(state)
            state, _, _, _ = agent.env.step((actions, targets))

        return agent

    def test_passive_env_fixed_seed_is_deterministic(self):
        # Arrange
        env1 = gym.make('SDSTests-GymEnvFixedSeedFixture-v0')
        env2 = gym.make('SDSTests-GymEnvFixedSeedFixture-v0')

        # Act
        self._passive_run_for(env1)
        self._passive_run_for(env2)

        # Assert
        self.assertListEqual(env1.unwrapped.sds_env.observation_space.graph.current_infected_nodes,
                             env2.unwrapped.sds_env.observation_space.graph.current_infected_nodes)
        self.assertListEqual(env1.unwrapped.sds_env.observation_space.graph.current_clear_nodes,
                             env2.unwrapped.sds_env.observation_space.graph.current_clear_nodes)
        self.assertAlmostEqual(env1.unwrapped.sds_env.observation_space.graph.g_.nodes[5]['immune'],
                               env2.unwrapped.sds_env.observation_space.graph.g_.nodes[5]['immune'])

    def test_passive_env_random_seed_is_stochastic(self):
        # Arrange
        env1 = gym.make('SDSTests-GymEnvRandomSeedFixture-v0')
        env2 = gym.make('SDSTests-GymEnvRandomSeedFixture-v0')

        # Act
        self._passive_run_for(env1)
        self._passive_run_for(env2)

        # Assert
        try:
            t1 = np.equal(np.array(env1.unwrapped.sds_env.observation_space.graph.current_infected_nodes),
                          np.array(env2.unwrapped.sds_env.observation_space.graph.current_infected_nodes))

            t2 = np.equal(env1.unwrapped.sds_env.observation_space.graph.current_clear_nodes,
                          env2.unwrapped.sds_env.observation_space.graph.current_clear_nodes)

            t3 = np.equal(env1.unwrapped.sds_env.observation_space.graph.g_.nodes[5]['immune'],
                          env2.unwrapped.sds_env.observation_space.graph.g_.nodes[5]['immune'])
            self.assertFalse(t1 & t2 & t3)
        except ValueError:
            # Failure to to length mismatch between lists in one of the tests. They can't be equal.
            pass

    def test_gym_env_with_dummy_agent(self):
        # Arrange
        agent = DummyAgent(env_spec='SDSTests-GymEnvFixedSeedFixture-v0')

        # Act
        self._active_run_for(agent)

        # Assert
        self.assertIsInstance(agent.env, gym.wrappers.TimeLimit)
        self.assertIsInstance(agent.env.unwrapped, GymEnv)
        self.assertEqual(agent.env_builder.env_spec, 'SDSTests-GymEnvFixedSeedFixture-v0')

    def test_gym_env_with_basic_agent(self):
        # Arrange
        agent = VaccinationAgent(env_spec='SDSTests-GymEnvFixedSeedFixture-v0')

        # Act
        self._active_run_for(agent)

        # Assert
        self.assertIsInstance(agent.env, gym.wrappers.TimeLimit)
        self.assertIsInstance(agent.env.unwrapped, GymEnv)
        self.assertEqual(agent.env_builder.env_spec, 'SDSTests-GymEnvFixedSeedFixture-v0')

    def test_gym_env_with_multi_agent(self):
        # Arrange
        agent = MultiAgent(name="Distancing, vaccination, treatment, masking",
                           env_spec='SDSTests-GymEnvFixedSeedFixture-v0',
                           agents=[DistancingPolicyAgent(actions_per_turn=15,
                                                         start_step={'isolate': 15, 'reconnect': 60},
                                                         end_step={'isolate': 55, 'reconnect': 80}),
                                   VaccinationPolicyAgent(actions_per_turn=5,
                                                          start_step={'vaccinate': 60},
                                                          end_step={'vaccinate': 80}),
                                   TreatmentPolicyAgent(actions_per_turn=5,
                                                        start_step={'treat': 50},
                                                        end_step={'treat': 80}),
                                   MaskingPolicyAgent(actions_per_turn=10,
                                                      start_step={'provide_mask': 40},
                                                      end_step={'provide_mask': 80})])

        # Act
        agent = self._active_run_for(agent)

        # Assert
        self.assertEqual(4, len(agent.agents))
        self.assertIsInstance(agent.env, gym.wrappers.TimeLimit)
        self.assertIsInstance(agent.env.unwrapped, GymEnv)
        self.assertEqual(agent.env_builder.env_spec, 'SDSTests-GymEnvFixedSeedFixture-v0')

    def test_gym_env_with_policy_agent(self):
        # Arrange
        agent = DistancingPolicyAgent(env_spec='SDSTests-GymEnvFixedSeedFixture-v0', actions_per_turn=4,
                                      start_step={'isolate': 10, 'reconnect': 50},
                                      end_step={'isolate': 40, 'reconnect': 60})

        # Act
        self._active_run_for(agent)

        # Assert
        self.assertIsInstance(agent.env, gym.wrappers.TimeLimit)
        self.assertIsInstance(agent.env.unwrapped, GymEnv)
        self.assertEqual(agent.env_builder.env_spec, 'SDSTests-GymEnvFixedSeedFixture-v0')

    def test_gym_env_with_rl_agent(self):
        # Arrange
        config_dict = RLKAgentConfigs(agent_name='flat_obs_dqn', env_spec='SDSTests-GymEnvFixedSeedFixture-v0',
                                      expected_obs_shape=(180,),
                                      env_wrappers=(partial(LimitObsWrapper, output=2),
                                                    FlattenObsWrapper),
                                      n_actions=5).build_for_dqn_untargeted()
        agent = DQNUntargeted(**config_dict)

        # Act
        self._active_run_for(agent)

        # Assert
        self.assertIsInstance(agent.env, FlattenObsWrapper)
        self.assertIsInstance(agent.env.unwrapped, GymEnv)
        self.assertEqual(agent.env_builder.env_spec, 'SDSTests-GymEnvFixedSeedFixture-v0')

    @unittest.skipUnless(int(f"{sys.version_info.major}{sys.version_info.minor}") > 36, 'deepcopy breaks in 3.6')
    def test_reset_matches_original_env(self):
        """env.reset() relies on sds_env cloning. This should return the original object. Make sure it does."""

        # Arrange
        env1 = gym.make('SDSTests-GymEnvFixedSeedFixture-v0')
        env2 = copy.deepcopy(env1)

        # Act
        _ = env1.reset()
        _ = env1.step(([], []))

        # Assert
        # gym env equality will not eval as equal
        self.assertNotEqual(env1, env2)
        # Should match to initial conditions
        self.assertEqual(env1.sds_env, env2.sds_env)
        # But not on history or changes by stepping
        self.assertNotEqual(env1.sds_env.history, env2.sds_env.history)
