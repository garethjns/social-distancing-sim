import abc

from social_distancing_sim.environment.environment import Environment


class TemplateBase(abc.ABC):
    @classmethod
    @abc.abstractmethod
    def build(cls) -> Environment:
        pass
