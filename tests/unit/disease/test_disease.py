import unittest

from social_distancing_sim.disease.disease import Disease


class TestDisease(unittest.TestCase):
    _sut = Disease

    def test_init_with_defaults(self):
        # Act
        disease = self._sut()

        # Assert
        self.assertIsInstance(disease, Disease)
        