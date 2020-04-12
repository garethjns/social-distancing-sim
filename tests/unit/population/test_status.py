import unittest

from social_distancing_sim.environment.status import Status


class TestStatus(unittest.TestCase):
    _sut = Status

    def test_init_with_defaults(self):
        # Act
        status = self._sut()

        # Assert
        self.assertIsInstance(status, Status)

    def test_eval_with_repr(self):
        # Arrange
        status = self._sut(infected=True, clear=False, immune=False, isolated=True)

        # Act
        status2 = eval(status.__repr__())

        # Assert
        self.assertEqual(status, status2)

    def test_init_with_invalid_infection_and_clear(self):
        # Actssert
        self.assertRaises(ValueError, lambda: Status(infected=True, clear=True))

    def test_init_with_invalid_infection_and_immune(self):
        # Actssert
        self.assertRaises(ValueError, lambda: Status(infected=True, immune=True))

    def test_deaths_init_voids_everything(self):
        # Act
        status = self._sut(alive=False)

        # Assert
        self.assertFalse(status.alive)
        self.assertFalse(status.infected)
        self.assertFalse(status.clear)
        self.assertFalse(status.immune)
        self.assertFalse(status.isolated)

    def test_death_post_init_voids_everything(self):
        # Arrange
        status = self._sut()

        # Act
        status.alive = False
        # Assert
        self.assertFalse(status.alive)
        self.assertFalse(status.infected)
        self.assertFalse(status.clear)
        self.assertFalse(status.immune)
        self.assertFalse(status.isolated)

    def test_infection_voids_clear_and_immune(self):
        # Arrange
        status = self._sut()

        # Act
        status.infected = True

        # Assert
        self.assertTrue(status.infected)
        self.assertFalse(status.clear)
        self.assertFalse(status.immune)

    def test_clear_to_false_voids_clear_and_immune(self):
        # Arrange
        status = self._sut()

        # Act
        status.clear = False

        # Assert
        self.assertTrue(status.infected)
        self.assertFalse(status.clear)
        self.assertFalse(status.immune)

    def test_clear_voids_infection_and_gives_clear(self):
        # Arrange
        status = self._sut()

        # Act
        status.clear = True

        # Assert
        self.assertTrue(status.clear)
        self.assertFalse(status.infected)

    def test_recovered_voids_infection_and_gives_clear(self):
        # Arrange
        status = self._sut()

        # Act
        status.recovered = True

        # Assert
        self.assertTrue(status.clear)
        self.assertFalse(status.infected)

    def test_infection_to_false_voids_infection_and_gives_clear(self):
        # Arrange
        status = self._sut()

        # Act
        status.infected = False

        # Assert
        self.assertTrue(status.clear)
        self.assertFalse(status.infected)
