from reinforcement_learning_keras.agents.q_learning.linear_q_agent import LinearQAgent

from social_distancing_sim.agent.learning_agent_base import LearningAgentBase


class LinearQUntargeted(LinearQAgent, LearningAgentBase):
    """Makes RLK LinearQaGENT compatible with SDS agent interface."""
    pass
