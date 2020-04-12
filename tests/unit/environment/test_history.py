import unittest
from typing import Callable

from social_distancing_sim.environment.history import History


class TestHistory(unittest.TestCase):
    _sut: Callable = History

    def setUp(self):
        self._hist = self._sut()

    def test_log_to_existing_key(self):
        # Arrange
        self._hist["new_key"] = [1]

        # Act
        self._hist.log({'new_key': 2})

        # Assert
        self.assertEqual(len(self._hist.keys()), 1)
        self.assertIsInstance(self._hist['new_key'], list)
        self.assertEqual(len(self._hist['new_key']), 2)
        self.assertEqual(self._hist['new_key'][0], 1)
        self.assertEqual(self._hist['new_key'][1], 2)

    def test_log_to_non_existing_key(self):
        # Act
        self._hist.log({'new_key2': 1})

        # Assert
        self.assertEqual(len(self._hist.keys()), 1)
        self.assertIsInstance(self._hist['new_key2'], list)
        self.assertEqual(len(self._hist['new_key2']), 1)
        self.assertEqual(self._hist['new_key2'][0], 1)
