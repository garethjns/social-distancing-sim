"""Run a single environments with perfect testing."""

import social_distancing_sim.environment as env

if __name__ == "__main__":
    # The graph is the "true" environments model, containing all the nodes and their data
    graph = env.Graph(
        community_n=50,
        community_size_mean=15,
        community_p_in=0.06,  # The likelihood of intra-community connections
        community_p_out=0.04,
    )  # The likelihood of inter-community connections

    # The ObservationSpace wraps the true graph to filter the available information about the Graph. Here
    # test_rate = 1 means the ObservationSpace has access to the full Graph.
    observation_space = env.ObservationSpace(
        graph=graph, test_rate=1  # Create environments graph and window into it
    )

    # Define a Disease with default parameters
    disease = env.Disease(name="COVID-19")

    # Define Healthcare availability with default settings
    healthcare = env.Healthcare()

    # Set the default plotting options, and add a second time-series plot to the figure showing turn score
    environment_plotting = env.EnvironmentPlotting(ts_fields_g2=["Turn score"])

    # Construct the environments
    pop = env.Environment(
        name="exps/example environments",
        disease=disease,
        healthcare=healthcare,
        environment_plotting=environment_plotting,
        observation_space=observation_space,
    )
    # Turn logging on
    pop.log_to_file = True

    # Run the environments, plotting and saving at each step
    pop.run(steps=150, plot=True, save=True)

    # Save .gif to './example environments/replay.gif'
    pop.replay()

    # History can be accessed in the History object. These keys can also be set to plot during the simulation in the
    # EnvironmentPlotting options
    print(pop.history.keys())
