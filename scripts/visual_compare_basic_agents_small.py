
"""
Run all the basic agents with a number of actions per turn (n reps = 1). Generate and save .gif.

Parameters here match the stats version run in scripts/stats_compare_basic_agents.py.
"""

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

    # Create a parameter set containing all combinations of the 4 basic agents, and a small set of n_actions
    agents = [agent.DummyAgent, agent.RandomAgent, agent.VaccinationAgent, agent.IsolationAgent]
    n_actions = [1, 3, 6]
    sims = []

    # Loop over the parameter set and create the Agents, Environments, and the Sim handler
    for n_act, agt in np.array(np.meshgrid(n_actions,
                                           agents)).T.reshape(-1, 2):
        agt_ = agt(actions_per_turn=n_act)

        # Name the environment according to the agent used
        env_ = env.Environment(name=f"{type(agt_).__name__} - {n_act} actions",
                               disease=env.Disease(name='COVID-19',
                                                   virulence=0.01,
                                                   seed=seed,
                                                   immunity_mean=0.95,
                                                   recovery_rate=0.9,
                                                   immunity_decay_mean=0.005),
                               healthcare=env.Healthcare(capacity=50),
                               environment_plotting=env.EnvironmentPlotting(ts_fields_g2=["Turn score", "Action cost",
                                                                                          "Overall score"]),
                               observation_space=env.ObservationSpace(
                                   graph=env.Graph(community_n=5,
                                                   community_size_mean=8,
                                                   seed=seed + 1),
                                   test_rate=1,
                                   seed=seed + 2),
                               initial_infections=15,
                               seed=seed + 3)

        sims.append(sim.Sim(env=env_,
                            agent=agt_,
                            n_steps=75,
                            plot=False,
                            save=True,
                            tqdm_on=True))  # Show progress bars for running sims

        # Run all the prepared Sims
        Parallel(n_jobs=4,
                 backend='loky')(delayed(run_and_replay)(sim) for sim in sims)

