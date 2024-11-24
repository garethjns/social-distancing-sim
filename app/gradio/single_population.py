import pathlib

import gradio as gr

from social_distancing_sim import environment as env


def run_single_population(
    name: str,
    steps: int,
    test_rate: float,
    community_n: int,
    community_size_mean: int,
    community_p_in: float = 0.06,
    community_p_out: float = 0.04,
    virulence: float = 0.005,
    recovery_rate: float = 0.99,
    immunity_mean: float = 0.66,
    immunity_decay_mean: float = 0.1,
    capacity: int = 200,
    max_penalty: float = 0.5,
    progress=gr.Progress(track_tqdm=True),
) -> pathlib.Path:
    # The graph is the "true" environments model, containing all the nodes and their data
    graph = env.Graph(
        community_n=community_n,
        community_size_mean=community_size_mean,
        # The likelihood of intra-community connections
        community_p_in=community_p_in,
        # The likelihood of inter-community connections
        community_p_out=community_p_out,
    )

    # The ObservationSpace wraps the true graph to filter the available information about the Graph. Here
    # test_rate = 1 means the ObservationSpace has access to the full Graph.
    observation_space = env.ObservationSpace(
        graph=graph, test_rate=test_rate  # Create environments graph and window into it
    )

    # Define a Disease with default parameters
    disease = env.Disease(
        name="COVID-19",
        virulence=virulence,
        recovery_rate=recovery_rate,
        immunity_mean=immunity_mean,
        immunity_decay_mean=immunity_decay_mean,
    )

    # Define Healthcare availability with default settings
    healthcare = env.Healthcare(
        capacity=capacity,
        max_penalty=max_penalty,
    )

    # Set the default plotting options, and add a second time-series plot to the figure showing turn score
    environment_plotting = env.EnvironmentPlotting(ts_fields_g2=["Turn score"])

    # Construct the environments

    pop = env.Environment(
        name=(
            name.lstrip("./").rstrip("/") if name is not None else "gradio_exps/example"
        ),
        disease=disease,
        healthcare=healthcare,
        environment_plotting=environment_plotting,
        observation_space=observation_space,
    )
    # Turn logging on
    pop.log_to_file = True

    # Run the environments, plotting and saving at each step
    pop.run(steps=steps, plot=True, save=True)

    # Render the output gif
    pop.replay()

    return pathlib.Path(f"{pop.environment_plotting.output_path}/replay.gif")
