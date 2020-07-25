import gym
import numpy as np


class LimitObsWrapper(gym.ObservationWrapper):
    """Limit the env's observation space to selected outputs."""

    def __init__(self, env: gym.Env, output: int) -> None:
        super().__init__(env)
        self.output = output
        # New env obs space shape
        self.observation_space = self.observation_space[self.output]

    def observation(self, obs: np.ndarray) -> np.ndarray:
        return obs[self.output].astype(int)
