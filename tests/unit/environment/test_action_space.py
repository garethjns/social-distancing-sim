import unittest
from social_distancing_sim.environment.action_space import ActionSpace


class TestActionSpace(unittest.TestCase):
    _sut = ActionSpace()

    def test_expected_default_actions_are_available(self):
        # Assert
        self.assertListEqual(['vaccinate', 'isolate', 'reconnect'], self._sut.available_actions)

    @unittest.skip(reason='TODO')
    def test_vaccinate_action(self):
        # TODO
        pass

    @unittest.skip(reason='TODO')
    def test_isolate_action(self):
        # TODO
        pass

    @unittest.skip(reason='TODO')
    def test_reconnect_action(self):
        # TODO
        pass
