"""Run all the basic policy agents with a number of actions per turn (n reps = 1). Generate and save .gif."""

from functools import partial

import numpy as np
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

    # Create a parameter set containing all combinations of the 3 policy agents, and a small set of n_actions
    agents = [agent.DummyAgent,
              partial(agent.DistancingPolicyAgent,
                      start_step={'isolate': 10,
                                  'reconnect': 50},
                      end_step={'isolate': 40,
                                'reconnect': 60}),
              partial(agent.VaccinationPolicyAgent,
                      start_step={'vaccinate': 20},
                      end_step={'vaccinate': 40}),
              partial(agent.TreatmentPolicyAgent,
                      start_step={'treat': 20},
                      end_step={'treat': 40}),
              ]
    n_actions = [10, 20]
    sims = []

    # Loop over the parameter set and create the Agents, Environments, and the Sim handler
    for n_act, agt in np.array(np.meshgrid(n_actions,
                                           agents)).T.reshape(-1, 2):
        agt_ = agt(actions_per_turn=n_act)

        # Name the environment according to the agent used
        env_ = env.Environment(name=f"{type(agt_).__name__} - {n_act} actions",
                               action_space=env.ActionSpace(isolate_efficiency=0.5,
                                                            vaccinate_efficiency=0.95),
                               disease=env.Disease(name='COVID-19',
                                                   virulence=0.008,
                                                   seed=seed,
                                                   immunity_mean=0.7,
                                                   recovery_rate=0.9,
                                                   immunity_decay_mean=0.01),
                               healthcare=env.Healthcare(capacity=75),
                               environment_plotting=env.EnvironmentPlotting(ts_fields_g2=["Actions taken",
                                                                                          "Overall score"]),
                               observation_space=env.ObservationSpace(
                                   graph=env.Graph(community_n=20,
                                                   community_size_mean=15,
                                                   considered_immune_threshold=0.7,
                                                   seed=seed + 1),
                                   test_rate=1,
                                   seed=seed + 2),
                               initial_infections=15,
                               seed=seed + 3)

        sims.append(sim.Sim(env=env_,
                            agent=agt_,
                            n_steps=300,
                            plot=False,
                            save=True,
                            tqdm_on=True))  # Show progress bars for running sims

    # Run all the prepared Sims
    Parallel(n_jobs=4,
             backend='loky')(delayed(run_and_replay)(sim) for sim in sims)
