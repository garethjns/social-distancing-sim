from functools import partial

import gym
from reinforcement_learning_keras.agents.components.helpers.virtual_gpu import VirtualGPU

from social_distancing_sim.agent.rl_agents.q_learning.dqn_untargeted import DQNUntargeted
from social_distancing_sim.agent.rl_agents.rlk_agent_configs import RLKAgentConfigs
from social_distancing_sim.environment.gym.wrappers.flatten_obs_wrapper import FlattenObsWrapper
from social_distancing_sim.environment.gym.wrappers.limit_obs_wrapper import LimitObsWrapper
from social_distancing_sim.sim import Sim


if __name__ == "__main__":
    gpu = VirtualGPU(gpu_memory_limit=2048,
                     gpu_device_id=0)

    gym.envs.register(id='SDS-746-v0',
                      entry_point='social_distancing_sim.environment.gym.environments.sds_746:SDS746',
                      max_episode_steps=1000)

    config_dict = RLKAgentConfigs(agent_name='flat_obs_dqn', env_spec='SDS-746-v0', expected_obs_shape=(746 * 6,),
                                  env_wrappers=(partial(LimitObsWrapper, output=2),
                                                FlattenObsWrapper),
                                  n_actions=5).build_for_dqn_untargeted()

    # Train agent
    agent = DQNUntargeted(**config_dict)
    agent.train(render=False, n_episodes=16)
    agent.save()

    # Eval
    env_spec = gym.make('SDS-746-v0').spec
    sim = Sim(env_spec=env_spec, agent=agent, n_steps=200, plot=True, save=True, tqdm_on=True,
              save_dir='exps/untargeted_dqn')
    sim.run()
