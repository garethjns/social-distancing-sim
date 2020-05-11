import warnings
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from social_distancing_sim.gym.agent.rl.epsilon import Epsilon
from social_distancing_sim.gym.agent.rl.q_learners.deep_q_agent import DeepQAgent
from social_distancing_sim.gym.gym_env import GymEnv
from social_distancing_sim.gym.wrappers.summary_graph_observation_wrapper import SummaryGraphObservationWrapper
from social_distancing_sim.templates.small import Small


def prepare_tf(memory_limit: int = 1024):
    import tensorflow as tf

    tf.config.experimental.set_virtual_device_configuration(tf.config.experimental.list_physical_devices('GPU')[0],
                                                            [tf.config.experimental.VirtualDeviceConfiguration(
                                                                memory_limit=memory_limit)])


def prepare(agent_gamma: float = 0.99,
            agent_eps: float = 0.99,
            agent_eps_decay: float = 0.001) -> Tuple[DeepQAgent, SummaryGraphObservationWrapper]:
    env = GymEnv(template=Small)
    env = SummaryGraphObservationWrapper(env)

    agent = DeepQAgent(env,
                       epsilon=Epsilon(initial=agent_eps,
                                       decay=agent_eps_decay),
                       gamma=agent_gamma)

    o1, o2 = agent.transform(env.observation_space.sample()[0:2])
    plt.plot(o1)
    plt.show()
    plt.imshow(o2.squeeze())
    plt.show()

    _, _ = agent.get_actions(env.observation_space.sample())

    return agent, env


def play_episode(env: SummaryGraphObservationWrapper, agent: DeepQAgent, max_episode_steps: int) -> float:
    obs = env.reset()
    done = False
    total_reward = 0
    step = 0
    while not done and (step < max_episode_steps):
        actions = agent.get_actions(obs)
        prev_obs = obs
        obs, reward, done, info = env.step(actions)

        agent.update(prev_obs, actions[0][0], reward, done, obs)

        step += 1

        # print(f"Took action: {action} @ eps: {agent.eps}, got reward: {reward}")
        total_reward += reward

    return total_reward


def train(agent: DeepQAgent, env: SummaryGraphObservationWrapper,
          n_episodes: int = 2000,
          max_episode_steps: int = 100) -> DeepQAgent:
    ep_rewards = []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        warnings.simplefilter("ignore", category=RuntimeWarning)

        for ep in tqdm(range(n_episodes)):
            total_reward = play_episode(env, agent, max_episode_steps)
            # print(f"Episode {ep} finished, reward = {total_reward}")
            ep_rewards.append(total_reward)
            print(total_reward)

            if not ep % 5:
                roll = 50
                plt.plot(np.convolve(ep_rewards, np.ones(roll), 'valid') / roll)
                plt.show()

    return agent


if __name__ == "__main__":
    prepare_tf(1024)

    agent_, env_ = prepare(agent_gamma=0.98,
                           agent_eps=0.95,
                           agent_eps_decay=0.002)
    agent_ = train(agent_, env_,
                   n_episodes=300,
                   max_episode_steps=200)

    agent_.save('deep_q_learner.pkl')
    DeepQAgent.load('deep_q_learner.pkl')
