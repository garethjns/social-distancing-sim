import gym
import numpy as np


class FlattenObsWrapper(gym.ObservationWrapper):
    """Limit the env's observation space to selected outputs."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        # New env obs space shape
        gym.spaces.box.Box(low=self.observation_space.low.min(), high=self.observation_space.high.max(),
                           shape=(self.observation_space.shape[0] * self.observation_space.shape[1],),
                           dtype=np.int16)

    def observation(self, obs: np.ndarray) -> np.ndarray:
        return obs.flatten()
