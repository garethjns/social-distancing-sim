# Social distancing simulator
![Testing is key](https://github.com/garethjns/social-distancing-sim/workflows/Testing%20is%20key/badge.svg)

Social distancing is an unfortunately unclear term; it means stay from other people to avoid killing yourself and them.

But why?

This package models disease spread through a population consisting of sub communities. These communities are inter and intra connected, the probability of these connections can be modified to model "normal" populations and populations exercising social distancing. Fewer connections mean fewer opportunities for disease to spread, slower progression through the population, a lower peak burden on the health care system, and fewer deaths..   

The code aims to be as simple and understandable as possible, but is still WIP (along with the documentation). See the Scripts/ folder for up to date usage examples.

![Example cats vs responsible](https://github.com/garethjns/social-distancing-sim/blob/master/images/joined.gif)   
([Discussion](https://new.reddit.com/r/dataisbeautiful/comments/fov56p/oc_comparing_the_effect_of_social_distancing_on/))

## Features and modelled components
 - [NetworkX](https://networkx.github.io/) graph-based population environment of inter and intra connected communities, where edge probabilities can model connected or socially distanced communities. Examples: **scripts/visual_compare_two_populations.py**, **scripts/visual_run_single_population.py**.
 - Disease virulence and imperfect and decaying immunity. Examples: **scripts/visual_compare_two_diseases_immunity**, **scripts/visual_compare_two_diseases_immunity_small.py**, **scripts/visual_compare_two_diseases_immunity.py**.
 - Healthcare capacity, effects on survival when overburdened
 - Test-driven observation space. Examples: **scripts/visual_compare_testing_rate.py**, **scripts/visual_compare_two_pop_testing_rate.py**.
 - Action space allowing for control strategies such as vaccination and isolation Examples: **scripts/visual_run_simulation_with_agent.py**, **scripts/visual_compare_basic_agents.py**.
 - Scoring system (WIP) for defining reinforcement agent rewards, ongoing economic costs, etc.
 - Visual simulation with history logging. Examples: **scripts/visual_*.py**.
 - Statistical simulation for multiple runs of the same parameters, aggregate statistics, experiment comparison (using [MLFlow](https://mlflow.org/)). Examples **scripts/stats_*.py**.

# Set up

````bash
pip install social_distancing_sim
````

# Run a simulation
See also scripts/run_single_population.py

````python
from social_distancing_sim.disease.disease import Disease
from social_distancing_sim.environment.environment import Environment
from social_distancing_sim.environment.observation_space import ObservationSpace
from social_distancing_sim.environment.graph import Graph
from social_distancing_sim.environment.healthcare import Healthcare

# Create a environment
pop = Environment(name="readme_example environment",
                 disease=Disease(name='COVID-19'),  # Define a disease
                 healthcare=Healthcare(),  # Define healthcare availability
                 observation_space=ObservationSpace(  # Create environment graph and window into it
                    graph=Graph(community_n=50,  # Number of communities
                                    community_size_mean=15, # Average number of people per community
                                    community_p_in=0.06,  # The likelihood of a connection between two community members
                                    community_p_out=0.04),  # The likelihood of a connection between two members of different communities. 
                    test_rate=1))  # Proportion of environment tested per step (testing determines observable data)

# Run the simulation for a number of days
pop.run(save=True, plot=True, steps=100)

# Save .gif to './example environment/replay.gif', duration sets time in s for each frame
pop.replay(duration=0.25)

# History can be accessed in the history attribute
pop.history.keys()
````

Or run with
````bash
python3 -m social-distancing-sim.scripts.run_single_population
````

# Compare two populations

```python
from social_distancing_sim.disease.disease import Disease
from social_distancing_sim.environment.environment import Environment
from social_distancing_sim.environment.observation_space import ObservationSpace
from social_distancing_sim.environment.graph import Graph
from social_distancing_sim.environment.healthcare import Healthcare

pop = Environment(name='A herd of cats',
                  disease=Disease(name='COVID-19'),
                  healthcare=Healthcare(),
                  observation_space=ObservationSpace(
                    graph=Graph(community_n=50,  
                                community_size_mean=15), 
                    test_rate=1)) 

pop_distanced = Environment(name='A socially responsible environment',
                            disease=Disease(name='COVID-19'),
                            healthcare=Healthcare(),
                            observation_space=ObservationSpace(
                              graph=Graph(community_n=50,
                                          community_size_mean=15,
                                          community_p_in=0.05,  # Reduced intra-community connections
                                           community_p_out=0.04),  # Reduced inter-community connections
                              test_rate=1)) 

pop_distanced.run(steps=200, plot=False, save=False).replay(duration=0.1)
pop.run(steps=200, plot=False, save=True).replay(duration=0.1)

```

Or run with
````bash
python3 -m social-distancing-sim.scripts.compare_two_populations
````


# Population dynamics

The dynamics of this simulation aim to be simple but interesting. They're built from reasoning and guesses at sensible values. They're not trained from data. The examples and default values aim to create simulations that do things in reasonable time, feel free to tweak them and see what happens.

Populations are randomly generated using a [networkx.random_partition_graph](https://networkx.github.io/documentation/stable/reference/generated/networkx.generators.community.random_partition_graph.html#networkx.generators.community.random_partition_graph). This creates a network consisting of communities where individual members have a given chance to be connected. Each individual member also has a lower chance to be connected to members of other communities.

In addition to communities, populations define a healthcare capacity. When above this capacity, the recovery rate from the disease is reduced.

The connections between individuals (graph nodes) define opportunities for a member to infect another. Each day (step) every infected node has one chance to infect each of it's neighbours, the chance of this happening is defined by the disease virulence.

Each day, infected nodes also have the chance to end their infection. The chance of this happening grows with the length of time the individual has been infected. If the infection ends, the individual either recovers and gains immunity, or dies. The chance of recovery is defined by the recovery rate of the disease, modified by the current burden on the healthcare system. When the heathcare system is below capacity, no penalty is applied to the recovery rate. When it's above, the recovery rate is reduced proportionally to the size of the burden. 

# Importance of testing

Run with
````bash
python3 -m social-distancing-sim.scripts.compare_two_pop_testing_rate
````

This script and figure compares the populations shown above, to their observable features given a set level of testing.

In the observed model, a proportion of the population (here ~4%) is tested per day the test validity is 5 days. Testing is conducted and observed status is updated as follows:
  - Each node has a specified chance of being tested per day - this chance is double for infected nodes and halved for clear nodes on the basis that people with symptoms are more likely to be tested.
  - If a node is tested and clear, status is marked as clear for the next 5 days
  - If a node is tested and is infected, status is marked as infected
  - For known infected nodes, status is switched to "dead" or "immune" at the end of disease, depending on survival
  - For unknown infected nodes, status is set to "dead" if they die, but left as unknown if they survive - this is based on the assumption we know about all deaths, but don't know about acquired immunity unless we know about infection.  

![Example testing rate](https://github.com/garethjns/social-distancing-sim/blob/master/images/joined_2.gif)  
([Discussion](https://new.reddit.com/r/dataisbeautiful/comments/fse6l1/oc_the_importance_of_testing_and_effect_on/))

# Immunity effects
Version 0.2.0 adds incomplete immunity and decay of immunity. These are part of the disease definition, and allow reinfection after a node has survived infection.

Run example with
````bash
python3 -m social-distancing-sim.scripts.compare_two_diseases_immunity
````  

![Example testing rate](https://github.com/garethjns/social-distancing-sim/blob/master/images/joined_3.gif)


# Basic agents and strategy comparison

This simulation compares how different responses, and when they applied, to a disease outbreak can affect the duration of the disease and number of deaths. 

The left column shows a mass vaccination response starting on day 21 (top) and day 40 (bottom). In this strategy 10 randomly selected uninfected nodes are vaccinated (ie. given 99% immunity) each day. Earlier activation of this strategy reduces infection peak and reduces deaths.  

The right hand column shows an isolation strategy, which is activated on day 25 (top) and day 40 (bottom). In this strategy, up to 8 randomly infected infected nodes are "isolated" per day - 95% of their edges are randomly removed. Edges are not re-added before the end of this simulation. Early adoption of this strategy is enough to totally stop the spread of disease.  

Later adoption of either strategy results in a similar outcomes, with greater peak infections and more deaths. One import thing to note here is that, although the early adoption of an isolation strategy performs best in this model set up, it relies on the rate of testing much more than the vaccination strategy - it needs the infected nodes to be identified wheres the vaccination strategy doesn't.  

![Example testing rate](https://github.com/garethjns/social-distancing-sim/blob/master/images/strat_comparison.gif)
