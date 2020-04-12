import unittest

from social_distancing_sim.environment.healthcare import Healthcare


class TestHealthcare(unittest.TestCase):
    _sut: Healthcare = Healthcare()

    def test_recovery_rate_penalty_is_0_below_capacity(self):
        # Act
        rrp = self._sut.recovery_rate_penalty(100)

        # Assert
        self.assertAlmostEqual(1, rrp)

    def test_recover_penalty_is_modified_just_above_capacity(self):
        # Act
        rrp = self._sut.recovery_rate_penalty(210)

        # Assert
        self.assertLess(rrp, 1)

    def test_recover_penalty_is_capped_above_capacity(self):
        # Act
        rrp = self._sut.recovery_rate_penalty(210000)

        # Assert
        self.assertAlmostEqual(0.5, rrp)
