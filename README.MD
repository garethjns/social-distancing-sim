# Social distancing simulator
![Testing is key](https://github.com/garethjns/social-distancing-sim/workflows/Testing%20is%20key/badge.svg)

Social distancing is an unfortunately unclear term; it means stay from other people to avoid killing yourself and them.

But why?

This package models disease spread through a population, allowing modification of many dynamics affecting spread. These simulations can be viewed as animations, or run many times to collect statistics. The simulation supports agent input, and can test the affect of policies such a mass vaccination and social distancing and isolation. Some examples are shown below.

The code aims to be as simple and understandable as possible, but is still WIP (along with the documentation). The documentation is mainly example driven see below and the Scripts/ folder for up to date usage examples.

# Population dynamics

The dynamics of this simulation aim to be simple but interesting, with scope in the parameters to run experiments on many different environment setups.

Populations are randomly generated using a [networkx.random_partition_graph](https://networkx.github.io/documentation/stable/reference/generated/networkx.generators.community.random_partition_graph.html#networkx.generators.community.random_partition_graph). This creates a network consisting of communities where individual members have a given chance to be connected. Each individual member also has a lower chance to be connected to members of other communities.

In addition to communities, populations define a healthcare capacity. When above this capacity, the recovery rate from the disease is reduced.

The connections between individuals (graph nodes) define opportunities for a member to infect another. Each day (step) every infected node has one chance to infect each of it's neighbours, the chance of this happening is defined by the disease virulence.

Each day, infected nodes also have the chance to end their infection. The chance of this happening grows with the length of time the individual has been infected. If the infection ends, the individual either recovers and gains immunity, or dies. The chance of recovery is defined by the recovery rate of the disease, modified by the current burden on the healthcare system. When the healthcare system is below capacity, no penalty is applied to the recovery rate. When it's above, the recovery rate is reduced proportionally to the size of the burden. 


## v0.4.0 Features and supported dynamics
 - [NetworkX](https://networkx.github.io/) graph-based population environment of inter and intra connected communities, where edge probabilities can model connected or socially distanced communities. Examples: **scripts/visual_compare_two_populations.py**, **scripts/visual_run_single_population.py**.
 - Disease virulence and imperfect and decaying immunity. Examples: **scripts/visual_compare_two_diseases_immunity**, **scripts/visual_compare_two_diseases_immunity_small.py**, **scripts/visual_compare_two_diseases_immunity.py**.
 - Healthcare capacity, effects on survival when overburdened
 - Test-driven observation space. Examples: **scripts/visual_compare_testing_rate.py**, **scripts/visual_compare_two_pop_testing_rate.py**.
 - Action space allowing for control strategies such as vaccination and isolation Examples: **scripts/visual_run_simulation_with_agent.py**, **scripts/visual_compare_basic_agents.py**.
 - Scoring system for quantifying outcomes and defining reinforcement agent rewards, ongoing economic costs, etc.
 - Visual simulation with history logging. Examples: **scripts/visual_*.py**.
 - Statistical simulation for multiple runs of the same parameters, aggregate statistics, experiment comparison (using [MLFlow](https://mlflow.org/)). Examples **scripts/stats_*.py**.
 - Basic (non-learning) agents to enact simple polices such as social distancing, vaccination, etc. 
  
## Planned features
 - Open AI gym API compatibility
 - Reinforcement learning agents
 - Less accurate testing, adding definable false positive and false negative rates
 - Docker container and rest API

# Set up

To use as a python package:
````bash
pip install social_distancing_sim
````

To get full repo and script examples to run:
````bash
git clone https://github.com/garethjns/social-distancing-sim
````

# Simulation structure and components
The social_distancing_sim package is split into 3 main modules; .sim, .environment, and .agent. See docstrings for object parameters and details.

## .environment
Contains the code for running the simulation, including the action space available to any agent. The top level object, Environment can be used run and plot individual simulations. Actions can be fed to the environment manually (or not at all), or can be handled by the Sim class in the .sim submodule (see below).

  - environment.**Environment** - Defines the environment as as a collection of objects:
    - .disease.**Disease** - Defines the disease; virulence, recovery rate, etc.
    - .healthcare.**HealthCare** - Defines the healthcare capacity of the population.
    - .history.**History** - Container for storing simulation stats and plotting time series.
    - environment_plotting.**EnvironmentPlotting** - Prepares and saves the main plot for each environment step, contains logic gor making simple animation from individual plots.
    - .scoring.**Scoring** - This object defines the points lost and gained for infections, deaths, clear node yield, etc.
    - .action_space.**ActionSpace** - The actions available within the environment that an agent can perform, and their costs.
    - .observation_space.**ObservationSpace** - Wrapper for the Graph object that handles testing and filters available data to external observers. Handles returning observed state in various ways.
      - .graph.**Graph** - The full simulation graph and graph plotting functionality
        - .status.**Status** - Each graph node consists of dictionary defining its true state, and a Status object containing it's accessible state. The Status class also defines the logic for ObservationState test updates. For example, if the ObservationState tests a node and finds it infected, Status.infected = True also automatically updates the dependent properties (such as "clear", "immune", etc.)

## .agent
Contains the code defining the agent interface and, currently, 4 basic agents.
  - agent_base.**AgentBase** - An abstract class agents should inherit from. Defines interface for action/target selection mechanisms. 
  - dummy_agent.**DummyAgent** - An agent that models the behaviour of many governments by not doing anything.
  - random_agent.**RandomAgent** - An agent that performs a number of random vaccination or isolation actions, with totally random target selection
  - isolation_agent.**IsolationAgent** - An agent that randomly isolates a number of infected + connected nodes and randomly reconnects recovered + isolated nodes.
  - vaccination_agent.**VaccinationAgent** - An agent that randomly vaccinates currently a number of non-infected nodes each turn.

## social_distancing_sim.sim
Contains objects to handle running and logging experiments with agent input
  - .sim.**Sim** - Handles the Environment, and an Agent. Steps the simulation, gets actions from agent, passes to env, etc. 
  - .multi_sim.**MultiSim** - Handles running Sim objects multiple times with different seeds. Outputs MLflow logs and aggregated statistics.
 

The simulated environment consists of a number of pparameterised objects.  

# Run a single simulation
![single simulation example](https://github.com/garethjns/social-distancing-sim/blob/master/images/single_simulation_example.gif)   
To run a single passive, visual, simulation, the Environment object can be defined and run without using the Sim and MultiSim handlers.

````bash
python3 -m social-distancing-sim.scripts.run_single_population
````

````python
import social_distancing_sim.environment as env
# The graph is the "true" population model, containing all the nodes and their data
graph = env.Graph(community_n=50,
                  community_size_mean=15,
                  community_p_in=0.06,  # The likelihood of intra-community connections
                  community_p_out=0.04)  # The likelihood of inter-community connections

# The ObservationSpace wraps the true graph to filter the available information about the Graph. Here
# test_rate = 1 means the ObservationSpace has access to the full Graph.
observation_space = env.ObservationSpace(graph=graph,  # Create environment graph and window into it
                                         test_rate=1)

# Define a Disease with default paramters
disease = env.Disease(name='COVID-19')

# Define Healthcare availability with default settings
healthcare = env.Healthcare()

# Set the default plotting options, and add a second time-series plot to the figure showing turn score
environment_plotting = env.EnvironmentPlotting(ts_fields_g2=["Turn score"])

# Construct the environment
pop = env.Environment(name="example environment",
                      disease=disease,
                      healthcare=healthcare,
                      environment_plotting=environment_plotting,
                      observation_space=observation_space)

# Run the environment, plotting and saving at each step
pop.run(steps=150,
        plot=True,
        save=True)

# Save .gif to './example environment/replay.gif'
pop.replay()

# History can be accessed in the History object. These keys can also be set to plot during the simulation in the 
# EnvironmentPlotting options 
print(pop.history.keys())
````

# Compare two populations: Social distancing
![Example cats vs responsible](https://github.com/garethjns/social-distancing-sim/blob/master/images/joined.gif)   
([Discussion](https://new.reddit.com/r/dataisbeautiful/comments/fov56p/oc_comparing_the_effect_of_social_distancing_on/))

The probabilities of inter and and intra community connectivity in the population can be modified to, for example, compare the effects between a densely connected population and a socially distanced population. In this example, fewer connections in the population give the disease fewer opportunities for disease to spread, which slows the progression through the population. This leads to a flatter infection curve and a lower peak burden on the health care system, results in fewer deaths.

````bash
python3 -m social-distancing-sim.scripts.compare_two_populations
````

```python
from joblib import Parallel, delayed

import social_distancing_sim.environment as env


def run_and_replay(pop, *args, **kwargs):
    pop.run(*args, **kwargs)
    if save:
        pop.replay()

save = True

# Create a population with high inter and intra connectivity
pop = env.Environment(name='A herd of cats',
                      disease=env.Disease(name='COVID-19'),
                      observation_space=env.ObservationSpace(graph=env.Graph(community_n=40,
                                                                             community_size_mean=16,
                                                                             seed=123),
                                                             test_rate=1),
                      environment_plotting=env.EnvironmentPlotting(ts_fields_g2=["Turn score"]))

# Create a population with reduced inter and intra connectivity
pop_distanced = env.Environment(name='A socially responsible environment',
                                disease=env.Disease(name='COVID-19'),
                                observation_space=env.ObservationSpace(graph=env.Graph(community_n=40,
                                                                                       community_size_mean=16,
                                                                                       community_p_in=0.05,
                                                                                       community_p_out=0.04,
                                                                                       seed=123),
                                                                       test_rate=1),
                                environment_plotting=env.EnvironmentPlotting(ts_fields_g2=["Turn score"]))

# Run and save both simulations in parallel
Parallel(n_jobs=2,
         backend='loky')(delayed(run_and_replay)(pop,
                                                 steps=300,
                                                 plot=False,
                                                 save=save) for pop in [pop, pop_distanced])
```





# Importance of testing: Modifying ObservationSpace test rate
![Example testing rate](https://github.com/garethjns/social-distancing-sim/blob/master/images/testing_example.gif)  
([Discussion](https://new.reddit.com/r/dataisbeautiful/comments/fse6l1/oc_the_importance_of_testing_and_effect_on/))

This script and figure compares the populations shown above, to their observable features given a set level of testing.

In the observed model, a proportion of the population (here ~4%) is tested per day the test validity is 5 days. Testing is conducted and observed status is updated as follows:
  - Each node has a specified chance of being tested per day - this chance is double for infected nodes and halved for clear nodes on the basis that people with symptoms are more likely to be tested.
  - If a node is tested and clear, status is marked as clear for the next 5 days
  - If a node is tested and is infected, status is marked as infected
  - For known infected nodes, status is switched to "dead" or "immune" at the end of disease, depending on survival
  - For unknown infected nodes, status is set to "dead" if they die, but left as unknown if they survive - this is based on the assumption we know about all deaths, but don't know about acquired immunity unless we know about infection.  

````bash
python3 -m social-distancing-sim.scripts.compare_two_pop_testing_rate
````

```python
from joblib import Parallel, delayed

import social_distancing_sim.environment as env


def run_and_replay(pop, *args, **kwargs):
    pop.run(*args, **kwargs)
    if save:
        pop.replay()

save = True

# Define the populations used in the social distancing comparison, but specify a test rate of 4% / per turn for
# the observation space. An plot will be added showing the data the observation space can see. 
pop_close = env.Environment(name='A herd of cats, observed',
                            disease=env.Disease(name='COVID-19'),
                            observation_space=env.ObservationSpace(graph=env.Graph(community_n=40,
                                                                                   community_size_mean=16,
                                                                                   seed=123),
                                                                   test_rate=0.04))

pop_distanced = env.Environment(name='A socially responsible environment, observed',
                                disease=env.Disease(name='COVID-19'),
                                observation_space=env.ObservationSpace(graph=env.Graph(community_n=40,
                                                                                       community_size_mean=16,
                                                                                       community_p_in=0.05,
                                                                                       community_p_out=0.04,
                                                                                       seed=123),
                                                                       test_rate=0.04))
# Run and save animations of both simulations
Parallel(n_jobs=2,
         backend='loky')(delayed(run_and_replay)(pop,
                                                 steps=130,
                                                 plot=False,
                                                 save=save) for pop in [pop_close, pop_distanced])
```


# Compare immunity effects
![Example testing rate](https://github.com/garethjns/social-distancing-sim/blob/master/images/joined_3.gif)

Version 0.2.0 adds incomplete immunity and decay of immunity. These are part of the disease definition, and allow reinfection after a node has survived infection.

````bash
python3 -m social-distancing-sim.scripts.compare_two_diseases_immunity
````  

```python
from joblib import Parallel, delayed

import social_distancing_sim.environment as env


def run_and_replay(pop, *args, **kwargs):
    pop.run(*args, **kwargs)
    if save:
        pop.replay(duration=duration)

save = True
duration = 0.1

# Create two environments, one containing a disease with imparts high immunity in survival, with a slow decay. The
# other with with a disease that imparts less immunity, which decays quicker.
pop_low_immunity = env.Environment(name="Low immunity environment",
                                   disease=env.Disease(name='COVID-19',
                                                       virulence=0.005,
                                                       recovery_rate=0.99,
                                                       immunity_mean=0.66,
                                                       immunity_decay_mean=0.1),
                                   environment_plotting=env.EnvironmentPlotting(
                                       ts_fields_g2=["Mean immunity (of immune nodes)",
                                                     "Mean immunity (of all alive nodes)"],
                                       ts_obs_fields_g2=["Known mean immunity (of immune nodes)",
                                                         "Known mean immunity (of all alive nodes)"]),
                                   observation_space=env.ObservationSpace(graph=env.Graph(community_n=50,
                                                                                          community_size_mean=15,
                                                                                          community_p_in=0.08,
                                                                                          community_p_out=0.04),
                                                                          test_rate=0.05),
                                   healthcare=env.Healthcare(capacity=200))

pop_high_immunity = env.Environment(name="High immunity environment",
                                    environment_plotting=env.EnvironmentPlotting(
                                        ts_fields_g2=["Mean immunity (of immune nodes)",
                                                      "Mean immunity (of all alive nodes)"],
                                        ts_obs_fields_g2=["Known mean immunity (of immune nodes)",
                                                          "Known mean immunity (of all alive nodes)"]),
                                    observation_space=env.ObservationSpace(graph=env.Graph(community_n=50,
                                                                                           community_size_mean=15,
                                                                                           community_p_in=0.08,
                                                                                           community_p_out=0.04),
                                                                           test_rate=0.05),
                                    healthcare=env.Healthcare(capacity=200))

Parallel(n_jobs=2,
         backend='loky')(delayed(run_and_replay)(pop,
                                                 steps=750,
                                                 plot=False,
                                                 save=True) for pop in [pop_low_immunity, pop_high_immunity])
```


# Basic agents and strategy comparison
![Test basic agents](https://github.com/garethjns/social-distancing-sim/blob/master/images/basic_agents_example.gif)

````bash
python3 -m social-distancing-sim.scripts.visual_compare_basic_agents
````  

```python
import numpy as np
from joblib import Parallel, delayed

import social_distancing_sim.agent as agent
import social_distancing_sim.environment as env
import social_distancing_sim.sim as sim


def run_and_replay(sim):
    sim.run()
    if sim.save:
        sim.env.replay()

seed = 123

# Create a parameter set containing all combinations of the 4 basic agents, and a small set of n_actions
agents = [agent.DummyAgent, agent.RandomAgent, agent.VaccinationAgent, agent.IsolationAgent]
n_actions = [3, 6, 12]
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
                               graph=env.Graph(community_n=15,
                                               community_size_mean=10,
                                               seed=seed + 1),
                               test_rate=1,
                               seed=seed + 2),
                           initial_infections=15,
                           seed=seed + 3)
    
    sims.append(sim.Sim(env=env_,
                        agent=agt_,
                        n_steps=125,
                        plot=False,
                        save=True,
                        tqdm_on=True))  # Show progress bars for running sims
    
# Run all the prepared Sims
Parallel(n_jobs=4,
         backend='loky')(delayed(run_and_replay)(sim) for sim in sims)


```

# MultiSims: Statistical comparisons - basic agents and strategy comparison

````bash
python3 -m social-distancing-sim.scripts.stats_compare_basic_agents
````  

````python
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm

import social_distancing_sim.agent as agent
import social_distancing_sim.environment as env
import social_distancing_sim.sim as sim


def plot_dists(result: str = "Overall score") -> plt.Figure:
    """Plot final score distributions across repetitions, for all agents."""
    fig, axs = plt.subplots(nrows=4,
                            ncols=1,
                            figsize=(8, 8))
    ax_map = {'DummyAgent': axs[0],
              'RandomAgent': axs[1],
              'VaccinationAgent': axs[2],
              'IsolationAgent': axs[3]}

    min_score = 0
    max_score = 0
    for run in multi_sims:
        min_score = min(min_score, run.results[result].min())
        max_score = max(max_score, run.results[result].max())

        sns.distplot(run.results[result],
                     hist=False,
                     ax=ax_map[type(run.sim.agent).__name__],
                     label=run.sim.agent.actions_per_turn)

    for ax_i, (ax_name, ax) in enumerate(ax_map.items()):
        ax.set_title(ax_name, fontweight='bold')
        ax.set_xlim([min_score - abs(min_score * 0.2), max_score + abs(max_score * 0.2)])
        if ax_i != 3:
            ax.set_xlabel('')
        else:
            ax.set_xlabel(ax.get_xlabel(),
                          fontweight='bold')
        ax.set_ylabel('Prop.',
                      fontweight='bold')
        ax.legend(title='n actions / turn')

    return fig

# For a specified parameter set....
agents = [agent.DummyAgent, agent.RandomAgent, agent.VaccinationAgent, agent.IsolationAgent]
n_actions = [3, 6, 12, 24]
multi_sims = []

# Create all Agent, Environments, Sims, and MultiSims. The MultiSim will run the Sim multiple times without
# visualisations
for n_act, agt in np.array(np.meshgrid(n_actions,
                                       agents)).T.reshape(-1, 2):
    agt_ = agt(actions_per_turn=n_act)
    env_ = env.Environment(name=f"{type(agt_).__name__} - {n_act} actions",
                           disease=env.Disease(name='COVID-19',
                                               virulence=0.01,
                                               seed=None,
                                               immunity_mean=0.95,
                                               recovery_rate=0.9,
                                               immunity_decay_mean=0.005),
                           healthcare=env.Healthcare(capacity=50),
                           observation_space=env.ObservationSpace(graph=env.Graph(community_n=15,
                                                                                  community_size_mean=10,
                                                                                  seed=None),
                                                                  test_rate=1,
                                                                  seed=None),
                           initial_infections=15,
                           seed=None)  # Use a different seed on every run)

    sim_ = sim.Sim(env=env_,
                   agent=agt_,
                   n_steps=125)

    multi_sims.append(sim.MultiSim(sim_,
                                   name='basic agent comparison 2',
                                   n_reps=100))  # Without plotting simulations run about 1000x faster!

# Run all the sims. No need to parallelize here as it's done across n reps in MultiSim.run()
for ms in tqdm(multi_sims):
    ms.run()

fig = plot_dists("Overall score")
plt.show()
fig.savefig('agent_comparison_score.png')

fig = plot_dists("Total deaths")
plt.show()
fig.savefig('agent_comparison_deaths.png')
````
