"""Run a single environment with perfect testing."""

from social_distancing_sim.environment.graph import Graph
from social_distancing_sim.environment.healthcare import Healthcare
from social_distancing_sim.environment.observation_space import ObservationSpace
from social_distancing_sim.environment.environment import Environment
from social_distancing_sim.disease.disease import Disease
from social_distancing_sim.environment.environment_plotting import EnvironmentPlotting

if __name__ == "__main__":
    save = True

    graph = Graph(community_n=50,
                  community_size_mean=15)

    pop = Environment(name="example environment",
                      disease=Disease(name='COVID-19'),
                      healthcare=Healthcare(),
                      observation_space=ObservationSpace(graph=graph,
                                                         test_rate=.2),
                      environment_plotting=EnvironmentPlotting(ts_fields_g2=["Score"],
                                                               ts_obs_fields_g2=["Observed score"]))

    pop.run(steps=150,
            plot=True,
            save=True)

    # Save .gif to './example environment/replay.gif'
    if save:
        pop.replay()
