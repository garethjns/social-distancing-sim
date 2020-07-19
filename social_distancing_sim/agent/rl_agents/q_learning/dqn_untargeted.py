from reinforcement_learning_keras.agents.q_learning.deep_q_agent import DeepQAgent

from social_distancing_sim.agent.learning_agent_base import LearningAgentBase


class DQNUntargeted(DeepQAgent, LearningAgentBase):
    """Makes RLK DeepQAgent compatible with SDS agent interface."""
