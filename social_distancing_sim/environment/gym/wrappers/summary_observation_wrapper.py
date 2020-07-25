from gym import ObservationWrapper


class SummaryObservationWrapper(ObservationWrapper):
    def observation(self, observation):
        return observation[0]
