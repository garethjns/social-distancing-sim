"""A number of different MultiAgent setups (n reps = 1). Generate and save .gif."""

import gym
import numpy as np
from joblib import Parallel, delayed

import social_distancing_sim.agent as agent
import social_distancing_sim.environment as env
import social_distancing_sim.sim as sim
from social_distancing_sim.environment.gym.gym_env import GymEnv
from social_distancing_sim.templates.template_base import TemplateBase


def run_and_replay(sim):
    sim.run()
    if sim.save:
        sim.env.replay()


STEPS = 250
MASKING_PARAMS = {
    "actions_per_turn": 10,
    "start_step": {"provide_mask": 40},
    "end_step": {"provide_mask": STEPS},
}
DISTANCING_PARAMS = {
    "actions_per_turn": 15,
    "start_step": {"isolate": 15, "reconnect": 60},
    "end_step": {"isolate": 55, "reconnect": STEPS},
}
VACCINATION_PARAMS = {
    "actions_per_turn": 5,
    "start_step": {"vaccinate": 60},
    "end_step": {"vaccinate": STEPS},
}
TREATMENT_PARAMS = {
    "actions_per_turn": 5,
    "start_step": {"treat": 50},
    "end_step": {"treat": STEPS},
}

# Create a parameter set containing all individual and pair combinations of the 4 policy agents,
# and non-exhaustive >2 combinations.
AGENTS = [
    agent.MultiAgent(
        name="Distancing", agents=[agent.DistancingPolicyAgent(**DISTANCING_PARAMS)]
    ),
    agent.MultiAgent(
        name="Vaccination", agents=[agent.VaccinationPolicyAgent(**VACCINATION_PARAMS)]
    ),
    agent.MultiAgent(
        name="Treatment", agents=[agent.TreatmentPolicyAgent(**TREATMENT_PARAMS)]
    ),
    agent.MultiAgent(
        name="Masking", agents=[agent.MaskingPolicyAgent(**MASKING_PARAMS)]
    ),
    # Pairs
    agent.MultiAgent(
        name="Distancing, vaccination",
        agents=[
            agent.DistancingPolicyAgent(**DISTANCING_PARAMS),
            agent.VaccinationPolicyAgent(**VACCINATION_PARAMS),
        ],
    ),
    agent.MultiAgent(
        name="Distancing, treatment",
        agents=[
            agent.DistancingPolicyAgent(**DISTANCING_PARAMS),
            agent.TreatmentPolicyAgent(**TREATMENT_PARAMS),
        ],
    ),
    agent.MultiAgent(
        name="Distancing, masking",
        agents=[
            agent.DistancingPolicyAgent(**DISTANCING_PARAMS),
            agent.MaskingPolicyAgent(**MASKING_PARAMS),
        ],
    ),
    agent.MultiAgent(
        name="Vaccination, treatment",
        agents=[
            agent.VaccinationPolicyAgent(**VACCINATION_PARAMS),
            agent.TreatmentPolicyAgent(**TREATMENT_PARAMS),
        ],
    ),
    agent.MultiAgent(
        name="Vaccination, masking",
        agents=[
            agent.VaccinationPolicyAgent(**VACCINATION_PARAMS),
            agent.MaskingPolicyAgent(**MASKING_PARAMS),
        ],
    ),
    agent.MultiAgent(
        name="Treatment, masking",
        agents=[
            agent.VaccinationPolicyAgent(**VACCINATION_PARAMS),
            agent.MaskingPolicyAgent(**MASKING_PARAMS),
        ],
    ),
    # Original triple
    agent.MultiAgent(
        name="Distancing, vaccination, treatment",
        agents=[
            agent.DistancingPolicyAgent(**DISTANCING_PARAMS),
            agent.VaccinationPolicyAgent(**VACCINATION_PARAMS),
            agent.TreatmentPolicyAgent(**TREATMENT_PARAMS),
        ],
    ),
    # All 4
    agent.MultiAgent(
        name="Distancing, vaccination, treatment, masking",
        agents=[
            agent.DistancingPolicyAgent(**DISTANCING_PARAMS),
            agent.VaccinationPolicyAgent(**VACCINATION_PARAMS),
            agent.TreatmentPolicyAgent(**TREATMENT_PARAMS),
            agent.MaskingPolicyAgent(**MASKING_PARAMS),
        ],
    ),
]


class EnvTemplate(TemplateBase):
    def build(self):
        seed = 125
        return env.Environment(
            name="visual_compare_multi_agents_custom_env",
            action_space=env.ActionSpace(
                isolate_efficiency=0.75,
                reconnect_efficiency=0.2,
                treatment_conclusion_chance=0.2,
                treatment_recovery_rate_modifier=1.8,
                vaccinate_efficiency=0.95,
            ),
            disease=env.Disease(
                name="COVID-19",
                virulence=0.0055,
                seed=seed,
                immunity_mean=0.7,
                recovery_rate=0.95,
                immunity_decay_mean=0.012,
            ),
            healthcare=env.Healthcare(capacity=200),
            environment_plotting=env.EnvironmentPlotting(
                auto_lim_x=False,
                ts_fields_g2=[
                    "Actions taken",
                    "Vaccinate actions",
                    "Isolate actions",
                    "Reconnect actions",
                    "Treat actions",
                    "Mask actions",
                ],
            ),
            observation_space=env.ObservationSpace(
                graph=env.Graph(
                    community_n=30,
                    community_size_mean=20,
                    community_p_out=0.08,
                    community_p_in=0.16,
                    seed=seed + 1,
                ),
                test_rate=1,
                seed=seed + 2,
            ),
            initial_infections=5,
            random_infection_chance=1,
            seed=seed + 3,
        )


class CustomEnv(GymEnv):
    template = EnvTemplate()


def run_agents():
    # Loop over the parameter set and create the Agents, Environments, and the Sim handler
    sims = []
    for agt in AGENTS:
        sims.append(
            sim.Sim(
                env_spec=gym.make(env_name).spec,
                agent=agt,
                n_steps=STEPS,
                plot=False,
                save=True,
                tqdm_on=True,
            )
        )  # Show progress bars for running sims

    # Run all the prepared Sims
    Parallel(n_jobs=60, backend="loky")(delayed(run_and_replay)(sim_) for sim_ in sims)


if __name__ == "__main__":
    # Prepare a custom environment
    env_name = f"SDSTests-CustomEnv{np.random.randint(2e6)}-v0"
    gym.envs.register(
        id=env_name,
        entry_point="scripts.visual_compare_multi_agents:CustomEnv",
        max_episode_steps=1000,
    )

    run_agents()
