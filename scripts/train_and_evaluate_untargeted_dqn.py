from functools import partial

import gym
import numpy as np
from reinforcement_learning_keras.agents.components.helpers.virtual_gpu import VirtualGPU

import social_distancing_sim.environment as env
from social_distancing_sim.agent.rl_agents.q_learning.dqn_untargeted import DQNUntargeted
from social_distancing_sim.agent.rl_agents.rlk_agent_configs import RLKAgentConfigs
from social_distancing_sim.environment import ActionSpace, EnvironmentPlotting
from social_distancing_sim.environment.gym.gym_env import GymEnv
from social_distancing_sim.environment.gym.wrappers.flatten_obs_wrapper import FlattenObsWrapper
from social_distancing_sim.environment.gym.wrappers.limit_obs_wrapper import LimitObsWrapper
from social_distancing_sim.sim import Sim
from social_distancing_sim.templates.template_base import TemplateBase


class EnvTemplate(TemplateBase):

    @classmethod
    def build(cls) -> env.Environment:
        return env.Environment(name="agent training example",
                               action_space=ActionSpace(),
                               environment_plotting=EnvironmentPlotting(
                                   ts_fields_g2=['Vaccinate actions completed', 'Isolate actions completed',
                                                 'Reconnect actions completed', 'Treat actions completed',
                                                 'Mask actions completed']),
                               disease=env.Disease(name='COVID-19',
                                                   virulence=0.006,
                                                   immunity_mean=0.6,
                                                   immunity_decay_mean=0.15),
                               healthcare=env.Healthcare(),
                               observation_space=env.ObservationSpace(graph=env.Graph(community_n=50,
                                                                                      community_size_mean=15,
                                                                                      community_p_in=0.1,
                                                                                      community_p_out=0.05,
                                                                                      seed=20200423),
                                                                      test_rate=1))


class CustomEnv(GymEnv):
    template = EnvTemplate()


if __name__ == "__main__":
    gpu = VirtualGPU(gpu_memory_limit=2048,
                     gpu_device_id=0)

    env_name = f"SDS-CustomEnv{np.random.randint(2e6)}-v0"
    gym.envs.register(id=env_name,
                      entry_point='scripts.train_and_evaluate_untargeted_dqn:CustomEnv',
                      max_episode_steps=1000)

    config_dict = RLKAgentConfigs(agent_name='flat_obs_dqn', env_spec=env_name, expected_obs_shape=(746 * 6,),
                                  env_wrappers=(partial(LimitObsWrapper, output=2),
                                                FlattenObsWrapper),
                                  n_actions=5).build_for_dqn_untargeted()

    # Train agent using rlk agents built in train function. Note that the agent only takes a single action per turn
    # unless the multiple actions wrapper is added. TODO: Add this wrapper for training but remove for future use.
    agent = DQNUntargeted(**config_dict)
    agent.train(render=False, n_episodes=25)
    agent.save()

    # Eval
    env_spec = gym.make(env_name).spec
    sim = Sim(env_spec=env_spec, agent=agent, n_steps=200, plot=False, save=True, tqdm_on=True, logging=True,
              save_dir='exps/untargeted_dqn')
    sim.run()
