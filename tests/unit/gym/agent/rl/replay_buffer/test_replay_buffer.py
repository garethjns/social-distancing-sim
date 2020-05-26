import unittest

import numpy as np

from social_distancing_sim.gym.agent.rl.replay_buffer import ReplayBuffer


class TesReplayBuffer(unittest.TestCase):
    def test_buffered_state_2_dims(self):
        rb = ReplayBuffer(replay_buffer_size=50, cache=3, state_dims=2)

        _ = [rb.append((np.zeros(shape=(5, 5)) + i, i, i, False)) for i in range(10)]

        ss, aa, rr, dd, ss_ = rb.sample(2)

        ss_0 = [np.unique(ss[0][0]), np.unique(ss[0][1]), np.unique(ss[0][2])]
        ss__0 = [np.unique(ss_[0][0]), np.unique(ss_[0][1]), np.unique(ss_[0][2])]

        self.assertEqual(aa[0], int(ss_0[0]))
        self.assertEqual(rr[0], int(ss_0[0]))

        self.assertListEqual([s + 1 for s in ss_0], ss__0)

    def test_buffered_state_1_dim(self):
        rb = ReplayBuffer(replay_buffer_size=50, cache=3, state_dims=1)

        _ = [rb.append((np.zeros(shape=5) + i, i, i, False)) for i in range(10)]

        ss, aa, rr, dd, ss_ = rb.sample(2)

        ss_0 = [np.unique(ss[0][0]), np.unique(ss[0][1]), np.unique(ss[0][2])]
        ss__0 = [np.unique(ss_[0][0]), np.unique(ss_[0][1]), np.unique(ss_[0][2])]

        self.assertEqual(aa[0], int(ss_0[0]))
        self.assertEqual(rr[0], int(ss_0[0]))
        self.assertListEqual([s + 1 for s in ss_0], ss__0)

    def test_unbuffered_state_1_dim(self):
        rb = ReplayBuffer(replay_buffer_size=50, cache=1, state_dims=1)

        _ = [rb.append((np.zeros(shape=(5)) + i, i, i, False)) for i in range(10)]

        ss, aa, rr, dd, ss_ = rb.sample(2)

        ss_0 = [np.unique(ss[0][0])]
        ss__0 = [np.unique(ss_[0][0])]

        self.assertEqual(aa[0], int(ss_0[0]))
        self.assertEqual(rr[0], int(ss_0[0]))
        self.assertListEqual([s + 1 for s in ss_0], ss__0)
