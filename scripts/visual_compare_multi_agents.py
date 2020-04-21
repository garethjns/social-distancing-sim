"""A number of different MultiAgent setups (n reps = 1). Generate and save .gif."""

from joblib import Parallel, delayed

import social_distancing_sim.agent as agent
import social_distancing_sim.environment as env
import social_distancing_sim.sim as sim


def run_and_replay(sim):
    sim.run()
    if sim.save:
        sim.env.replay()


if __name__ == "__main__":
    seed = 123
    steps = 250
    distancing_params = {"actions_per_turn": 15,
                         "start_step": {'isolate': 15, 'reconnect': 60},
                         "end_step": {'isolate': 55, 'reconnect': steps}}
    vaccination_params = {"actions_per_turn": 5,
                          "start_step": {'vaccinate': 60},
                          "end_step": {'vaccinate': steps}}
    treatment_params = {"actions_per_turn": 5,
                        "start_step": {'treat': 50},
                        "end_step": {'treat': steps}}

    # Create a parameter set containing all combinations of the 3 policy agents, and a small set of n_actions
    agents = [agent.MultiAgent(name="Distancing",
                               agents=[agent.DistancingPolicyAgent(**distancing_params)]),
              agent.MultiAgent(name="Vaccination",
                               agents=[agent.VaccinationPolicyAgent(**vaccination_params)]),
              agent.MultiAgent(name="Treatment",
                               agents=[agent.TreatmentPolicyAgent(**treatment_params)]),
              agent.MultiAgent(name="Distancing, vaccination",
                               agents=[agent.DistancingPolicyAgent(**distancing_params),
                                       agent.VaccinationPolicyAgent(**vaccination_params)]),
              agent.MultiAgent(name="Distancing, treatment",
                               agents=[agent.DistancingPolicyAgent(**distancing_params),
                                       agent.TreatmentPolicyAgent(**treatment_params)]),
              agent.MultiAgent(name="Vaccination, treatment",
                               agents=[agent.VaccinationPolicyAgent(**vaccination_params),
                                       agent.TreatmentPolicyAgent(**treatment_params)]),
              agent.MultiAgent(name="Distancing, vaccination, treatment",
                               agents=[agent.DistancingPolicyAgent(**distancing_params),
                                       agent.VaccinationPolicyAgent(**vaccination_params),
                                       agent.TreatmentPolicyAgent(**treatment_params)])]

    # Loop over the parameter set and create the Agents, Environments, and the Sim handler
    sims = []
    for agt in agents:
        # Name the environment according to the agent used
        env_ = env.Environment(name=f"{type(agt).__name__} - {agt.name}",
                               action_space=env.ActionSpace(isolate_efficiency=0.75,
                                                            reconnect_efficiency=0.2,
                                                            treatment_conclusion_chance=0.2,
                                                            treatment_recovery_rate_modifier=1.8,
                                                            vaccinate_efficiency=0.95),
                               disease=env.Disease(name='COVID-19',
                                                   virulence=0.0055,
                                                   seed=seed,
                                                   immunity_mean=0.7,
                                                   recovery_rate=0.95,
                                                   immunity_decay_mean=0.012),
                               healthcare=env.Healthcare(capacity=200),
                               environment_plotting=env.EnvironmentPlotting(
                                   auto_lim_x=False,
                                   ts_fields_g2=["Actions taken", "Vaccinate actions", "Isolate actions",
                                                 "Reconnect actions", "Treat actions"]),
                               observation_space=env.ObservationSpace(
                                   graph=env.Graph(community_n=30,
                                                   community_size_mean=20,
                                                   community_p_out=0.08,
                                                   community_p_in=0.16,
                                                   seed=seed + 1),
                                   test_rate=1,
                                   seed=seed + 2),
                               initial_infections=5,
                               random_infection_chance=1,
                               seed=seed + 3)

        sims.append(sim.Sim(env=env_,
                            agent=agt,
                            n_steps=steps,
                            plot=False,
                            save=True,
                            tqdm_on=True))  # Show progress bars for running sims

    # Run all the prepared Sims
    Parallel(n_jobs=9,
             backend='loky')(delayed(run_and_replay)(sim) for sim in sims)
