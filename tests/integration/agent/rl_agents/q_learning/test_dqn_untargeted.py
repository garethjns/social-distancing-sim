import glob
import os
import tempfile
import unittest
from functools import partial
from unittest.mock import patch

import gym
from reinforcement_learning_keras.agents.components.helpers.virtual_gpu import VirtualGPU

from social_distancing_sim.agent.rl_agents.q_learning.dqn_untargeted import DQNUntargeted
from social_distancing_sim.agent.rl_agents.rlk_agent_configs import RLKAgentConfigs
from social_distancing_sim.environment.gym.wrappers.flatten_obs_wrapper import FlattenObsWrapper
from social_distancing_sim.environment.gym.wrappers.limit_obs_wrapper import LimitObsWrapper
from social_distancing_sim.sim import Sim, MultiSim
from tests.common.env_fixtures import register_test_envs


class TestDQNUntargeted(unittest.TestCase):
    _sut = DQNUntargeted
    _env = 'SDSTests-GymEnvFixedSeedFixture-v0'
    _png_path: str = "graphs/*.png"

    @classmethod
    def setUpClass(cls):
        cls.gpu = VirtualGPU(gpu_memory_limit=4096,
                             gpu_device_id=0)

        register_test_envs()

    def setUp(self):
        self._built_env = gym.make(self._env)
        self._tmp_dir = tempfile.TemporaryDirectory()

        config_dict = RLKAgentConfigs(agent_name=f'{self._tmp_dir.name}/flat_obs_dqn', env_spec=self._env,
                                      expected_obs_shape=(180,),
                                      env_wrappers=(partial(LimitObsWrapper, output=2),
                                                    FlattenObsWrapper),
                                      n_actions=5).build_for_dqn_untargeted()

        self._agent = self._sut(**config_dict)

    def test_train_agent_with_train_method(self):
        # Act
        self._agent.train(render=False, n_episodes=1)

    @patch("social_distancing_sim.environment.environment_plotting.plt.show")
    def test_run_agent_in_sim_render_on(self, *args):
        # Arrange
        self._agent.train(render=False, n_episodes=1)

        # Act
        sim = Sim(env_spec=self._built_env.spec, agent=self._agent, n_steps=3, plot=True, save=True, tqdm_on=True,
                  save_dir=f'{self._tmp_dir.name}/test_agent_sim_render_on')
        sim.run()

        # Assert
        self.assertEqual(0, len(glob.glob(os.path.join(self._tmp_dir.name, self._png_path))))

    def test_run_agent_in_sim_render_off(self):
        # Arrange
        self._agent.train(render=False, n_episodes=1)

        # Act
        sim = Sim(env_spec=self._built_env.spec, agent=self._agent, n_steps=3, plot=False, save=False, tqdm_on=True,
                  save_dir=f'{self._tmp_dir.name}/test_agent_sim_render_off')
        sim.run()

        # Assert
        self.assertEqual(0, len(glob.glob(os.path.join(self._tmp_dir.name, self._png_path))))

    @patch("social_distancing_sim.environment.environment_plotting.plt.show")
    def test_run_agent_without_sim_render_on(self, *args):
        # Arrange
        self._agent.train(render=False, n_episodes=1)
        self._built_env.sds_env.environment_plotting.set_output_path(self._tmp_dir.name)
        self._agent.attach_to_env(self._built_env)

        # Act
        self._agent.play_episode(render=True, max_episode_steps=3)
        self._agent.env.replay()

        # Assert
        self.assertEqual(3, len(glob.glob(os.path.join(self._tmp_dir.name, self._png_path))))

    def test_run_agent_without_sim_render_off(self):
        # Arrange
        self._agent.train(render=False, n_episodes=1)
        self._built_env.sds_env.environment_plotting.set_output_path(self._tmp_dir.name)
        self._agent.attach_to_env(self._built_env)

        # Act
        self._agent.play_episode(render=False, max_episode_steps=3)

        # Assert
        self.assertEqual(0, len(glob.glob(os.path.join(self._tmp_dir.name, self._png_path))))

    def test_in_multisim_single_job(self):
        # Arrange
        self._agent.train(render=False, n_episodes=1)
        self._agent.save()

        # Act
        ms = MultiSim(Sim(env_spec=self._built_env.spec, tqdm_on=False, agent=self._agent, n_steps=3),
                      n_jobs=1, n_reps=3)
        ms.run()
