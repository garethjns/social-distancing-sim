import unittest
from social_distancing_sim.agent.multi_agent.multi_agent import MultiAgent
from social_distancing_sim.agent.basic_agents.isolation_agent import IsolationAgent
from social_distancing_sim.agent.basic_agents.treatment_agent import TreatmentAgent
from social_distancing_sim.agent.basic_agents.vaccination_agent import VaccinationAgent
from social_distancing_sim.agent.policy_agents.distancing_policy_agent import DistancingPolicyAgent
from social_distancing_sim.agent.policy_agents.treatment_policy_agent import TreatmentPolicyAgent
from social_distancing_sim.agent.policy_agents.vaccination_policy_agent import VaccinationPolicyAgent


class TestMultiAgent(unittest.TestCase):
    """TODO: WIP"""

    def setUp(self):
        self.distancing_policy_params = {"actions_per_turn": 15,
                                         "start_step": {'isolate': 15, 'reconnect': 60},
                                         "end_step": {'isolate': 55, 'reconnect': 100}}
        self.vaccination_policy_params = {"actions_per_turn": 5,
                                          "start_step": {'vaccinate': 60},
                                          "end_step": {'vaccinate': 100}}
        self.treatment_policy_params = {"actions_per_turn": 5,
                                        "start_step": {'treat': 50},
                                        "end_step": {'treat': 100}}

    def test_create_with_3_basic_agents(self):
        MultiAgent(name="Isolation, vaccination, treatment",
                   agents=[IsolationAgent(),
                           TreatmentAgent(),
                           VaccinationAgent()])

    def test_create_with_3_policy_agents(self):
        MultiAgent(name="Distancing, vaccination, treatment",
                   agents=[DistancingPolicyAgent(**self.distancing_policy_params),
                           VaccinationPolicyAgent(**self.vaccination_policy_params),
                           TreatmentPolicyAgent(**self.treatment_policy_params)])
