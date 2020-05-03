from gym import ObservationWrapper


class SummaryGraphObservationWrapper(ObservationWrapper):
    def observation(self, observation):
        return observation[0:2]
