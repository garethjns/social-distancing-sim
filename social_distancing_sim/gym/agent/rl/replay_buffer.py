import collections
from dataclasses import dataclass
from typing import Any
from typing import Tuple

import numpy as np


@dataclass
class ReplayBuffer:
    """Simple replay buffer non-redundant state."""
    replay_buffer_size: int = 200
    state_dims: int = 1
    cache: int = 0

    def __post_init__(self):
        self._state_queue = collections.deque(maxlen=self.replay_buffer_size + self.cache + 1)
        self._other_queue = collections.deque(maxlen=self.replay_buffer_size + self.cache + 1)

        self.queue = collections.deque(maxlen=self.replay_buffer_size)

        # If not caching, remove a dimension
        if self.cache == 1:
            self.state_dims -= 1

    def __len__(self):
        return self.n if self.n > 0 else 0

    @property
    def n(self) -> int:
        return len(self._state_queue) - self.cache - 1

    def append(self, items: Tuple[Any, int, float, bool]):
        """
        :param items: (s, a, r, s_, d) where (n x t, 1 x t, 1 x t, n x t, 1 x t)
        """
        self._state_queue.append(items[0])
        self._other_queue.append(items[1::])

    def sample(self, n: int):
        if n > self.n:
            raise ValueError

        idx = np.random.randint(0, self.n, n)
        if self.cache > 0:
            # Add a dimensions to stack on
            ss = [np.stack([self._state_queue[i + c] for c in range(self.cache)]).squeeze()
                  for i in idx]
            ss_ = [np.stack([self._state_queue[i + c + 1] for c in range(self.cache)]).squeeze()
                   for i in idx]
        else:
            # Skp the cache dimension
            ss = [self._state_queue[i] for i in idx]
            ss_ = [self._state_queue[i + 1] for i in idx]

        ard = [self._other_queue[i] for i in idx]
        aa = [a for (a, _, _) in ard]
        rr = [r for (_, r, _) in ard]
        dd = [d for (_, _, d) in ard]

        return ss, aa, rr, dd, ss_
