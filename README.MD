# Social distancing simulator
![Testing is key](https://github.com/garethjns/social-distancing-sim/workflows/Testing%20is%20key/badge.svg) [![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=garethjns_social-distancing-sim&metric=alert_status)](https://sonarcloud.io/dashboard?id=garethjns_social-distancing-sim)

This package models disease spread through a population, allowing modification of many dynamics affecting spread. The simulations can be viewed as animations, or run many times to collect statistics, evaluate response strategies, etc. The simulation supports agent input, which can either enact scripted policies such as mass vaccination and social distancing, or reinforcement learning agents that have learned their own strategies through experience.

The code aims to be as simple and understandable as possible, but is still WIP (along with the documentation) so expect things to break and change. The documentation is mainly example driven see [below](#usage_example) and the Scripts/ folder for up to date usage examples. 

![Example cats vs responsible](https://github.com/garethjns/social-distancing-sim/blob/master/images/joined.gif)   

# Contents
1. [Dynamics](#dynamics)
2. [Features](#features)
3. [Set up](#set_up)
4. [Usage example](#usage_example)
5. [Package structure and components](#package_structure)
6. [Experiments](#experiments)
    1. [Social distancing](#social_distancing)
    2. [Using masks](#using_masks)
    3. [Importance of testing](#importance_of_testing)
    4. [Herd immunity](#herd_immunity)
    5. [Basic agents](basic_agents)
    6. [Static policies](#static_policies)
    7. [Reinforcement learning agents](#reinforcement_agents)


# Dynamics <a name="dynamics"></a>
## Simulation <a name="simulation"></a>

The dynamics of this simulation aim to be simple but interesting, with scope in the parameters to run experiments on many different environment setups.

Populations are randomly generated using a [networkx.random_partition_graph](https://networkx.github.io/documentation/stable/reference/generated/networkx.generators.community.random_partition_graph.html#networkx.generators.community.random_partition_graph). This creates a network consisting of communities where individual members have a given chance to be connected. Each individual member also has a lower chance to be connected to members of other communities.

The connections between individuals (graph nodes) define opportunities for a member to infect another. Each day (step) every infected node has one chance to infect each of its neighbours, the chance of this happening is defined by the disease virulence and modified by interacting factors such as immunity, and use of masks by the target or source.

Each day, infected nodes also have the chance to end their infection. The probability of this happening grows with the length of time the individual has been infected. If the infection ends, the individual either recovers and gains immunity, or dies. The chance of recovery is defined by the recovery rate of the disease, modified by the current burden on the healthcare system. When the healthcare system is below capacity, no penalty is applied to the recovery rate. When it's above, the recovery rate is reduced proportionally to the size of the burden. If a node survives, it gains (or not) imperfect immunity that decays with time. 

In addition to communities, populations define a healthcare capacity. When above this capacity, the recovery rate from the disease is reduced.


## Agents <a name="agents"></a>

The simulation environment defines an action space that allows agents to perform actions each turn and influence disease spread. This interface supports basic agents social_distancing_sim.agent), "policy" agents with hardcoded logic, and reinforcement learning agents (supporting the OpenAI Gym API).

Agents are able to perform treatment, isolate, reconnect and vaccinate actions. Basic agents typically perform single actions in a semi-targeted fashion, and "policy" agents support multiple basic agents operating over different time periods. This allows for definition and experimentation with different strategies for managing outbreaks. (Note here "policy" refers to scripted strategy like isolating early, vaccinating when available, reconnecting nodes later on, etc. rather than a reinforcement learning agents learned policy).  
 
A flexible scoring system allows for setting of action costs and environment rewards and penalties. This can be used for agent/policy evaluation, and for training of the included RL agents (social_distancing_sim.gym.agent)


# v0.8.0 Features<a name="features"></a>
 - [NetworkX](https://networkx.github.io/) graph-based population environment of inter and intra connected communities, where edge probabilities can model connected or socially distanced communities. Examples: **scripts/visual_compare_two_populations.py**, **scripts/visual_run_single_population.py**.
 - Disease virulence and imperfect and decaying immunity. Examples: **scripts/visual_compare_two_diseases_immunity**, **scripts/visual_compare_two_diseases_immunity_small.py**, **scripts/visual_compare_two_diseases_immunity.py**.
 - Healthcare capacity, effects on survival when overburdened
 - Test-driven observation space. Examples: **scripts/visual_compare_testing_rate.py**, **scripts/visual_compare_two_pop_testing_rate.py**.
 - Action space allowing for control strategies such as vaccination, isolation, provision of masks, etc. Examples: **scripts/visual_run_simulation_with_agent.py**, **scripts/visual_compare_basic_agents.py**.
 - Scoring system for quantifying outcomes and defining reinforcement agent rewards, ongoing economic costs, etc.
 - Visual simulation with history logging. Examples: **scripts/visual_*.py**.
 - Statistical simulation for multiple runs of the same parameters, aggregate statistics, experiment comparison (using [MLFlow](https://mlflow.org/)). Examples **scripts/stats_*.py**.
 - Basic (non-learning) agents to enact simple polices such as social distancing, vaccination, etc. 
 - Open AI Gym compatibility
 - Linear and deep-q reinforcement learning agents Examples: **scripts/train_deep_q_learner.py, scripts/train_linear_q_learner.py**.
 - A scoring system with settable action costs and environment rewards/penalties 
  
### Planned
 - Agents supporting specific node targeting.
 - Less accurate testing, adding definable false positive and false negative rates
 - Docker container and rest API

# Running in Docker
```bash
docker pull garethjns/social-distancing-sim
docker run -p 8000:8000 social-distancing-sim
```

For now, REST API supports get requests for running a few examples. These return either the rendered gif, or the stats for the simulation (JSON). See http://localhost:8001/docs for all implemented methods.

For example (these will probably take a while to run):  
http://localhost:8000/run -> outputs gif  
http://localhost:8000/run_visual -> outputs json (if running through a browser, Firefox does a better job of displaying this than Chrome).  

The number of steps run and environment template to run can be specified in the url, eg.
http://localhost:8000/run?env=SDS-746-v0&steps=300


# Set up <a name="set_up"></a>

To use as a python package:
````bash
pip install social_distancing_sim
````

To get full repo and script examples to run:
````bash
git clone https://github.com/garethjns/social-distancing-sim
````

# Usage examples <a name="usage_example"></a>

## Simple simulation
![single simulation example](https://github.com/garethjns/social-distancing-sim/blob/master/images/single_simulation_example.gif)   
To run a single passive, visual, simulation, the Environment object can be defined and run without using the Sim and MultiSim handlers.

```bash
python3 -m scripts.run_single_population
```

Create environment components. See [below](#package_structure) for a more information on these. 
```python
import social_distancing_sim.environment as env
# The graph is the "true" environment model, containing all the nodes and their data
graph = env.Graph(community_n=50,
                  community_size_mean=15,
                  community_p_in=0.06,  # The likelihood of intra-community connections
                  community_p_out=0.04)  # The likelihood of inter-community connections

# The ObservationSpace wraps the true graph to filter the available information about the Graph. Here
# test_rate = 1 means the ObservationSpace has access to the full Graph.
observation_space = env.ObservationSpace(graph=graph,  # Create environments graph and window into it
                                         test_rate=1)

# Define a Disease with default parameters
disease = env.Disease(name='COVID-19')

# Define Healthcare availability with default settings
healthcare = env.Healthcare()

# Set the default plotting options, and add a second time-series plot to the figure showing turn score
environment_plotting = env.EnvironmentPlotting(ts_fields_g2=["Turn score"])
```

Construct the environment
```Python
pop = env.Environment(name="example environments",
                      disease=disease,
                      healthcare=healthcare,
                      environment_plotting=environment_plotting,
                      observation_space=observation_space)

# To log to file, turn on manually. This will be created in [env_name]/log.txt
pop.log_to_file = True
```

Run the simulation
```Python
# Run the environments, plotting and saving at each step. Frames are saved to [env_name]/graphs/[step]_graph.png
pop.run(steps=150, plot=True, save=True)

# Save .gif of the frames to '[env_name]/replay.gif'
pop.replay()

# History can be accessed in the History object. These keys can also be set to plot during the simulation in the 
# EnvironmentPlotting options 
print(pop.history.keys())
```

## Simulation with an agent
The Sim class handles running environments and agents together. It uses the OpenAI Gym environment interface. It'll also
run without any agent specified (in such a case it'll use a DummyAgent that does nothing.)

```bash
python3 -m scripts.visual_run_simulation_with_agent
```

Define an environment template that will be used to construct the environment for the agent.
````python
from social_distancing_sim.templates.template_base import TemplateBase
import social_distancing_sim.environment as env
from social_distancing_sim.environment.gym.gym_env import GymEnv

# Define a template for the environment. This can include set seeds.
class EnvTemplate(TemplateBase):
    def build(self):
        return env.Environment(name="example_sim_env",
                               action_space=env.ActionSpace(),
                               disease=env.Disease(),
                               healthcare=env.Healthcare(),
                               environment_plotting=env.EnvironmentPlotting(ts_fields_g2=["Actions taken",
                                                                                          "Overall score"]),
                               observation_space=env.ObservationSpace(
                                   graph=env.Graph(community_n=20,
                                                   community_size_mean=15,
                                                   considered_immune_threshold=0.7),
                                   test_rate=1),
                               initial_infections=15)

# Define an gym compatible environment environment 
class CustomEnv(GymEnv):
    template = EnvTemplate()
````

Register the environment with gym. Note that here, the entry point is really pointing to one of the predefined templates, rather than the one defined in the example above. 

This should be changed to point to the correct env as appropriate. It can point to the same file the env is defined in.
 
````python
import gym

env_name = f"SDSTests-CustomEnvForReadme-v0"
gym.envs.register(id=env_name, entry_point='social_distancing_sim.environment.gym.environments.sds_746:SDS746',
                  max_episode_steps=1000)
env_spec = gym.make(env_name).spec
````

Create the simulation, and specify the agent (can be None).
```python
import social_distancing_sim.sim as sim
from social_distancing_sim.agent.basic_agents.vaccination_agent import VaccinationAgent

# Create Sim
sim_ = sim.Sim(env_spec=env_spec,
              agent=VaccinationAgent(actions_per_turn=15),
              plot=True,  # Show plots during simulation 
              save=True,  # Save plots after showing
              logging=True,  # Log to file
              tqdm_on=True)  # Show tqdm waitbar

sim_.run()
```

## Training / simulation with an rl agent
This is WIP, but see scripts/train_and_evaluate_untargeted_dqn.py. Aim is to work with the same Sim interface for evaluation. 

## MultiSims
Run a Sim multiple times and return stats. This will handle multiprocessing, rebuilding environments, transporting agents, etc., and logging results to MLflow.

Register an Env (if required) and define Sim as above but leave logging, plotting, and saving of plots off!. It makes sense to use an environment that isn't totally deterministic. Each agent and environment component has settable seeds, although they default to None.


```Python
import gym
from social_distancing_sim.agent.basic_agents.vaccination_agent import VaccinationAgent
import social_distancing_sim.sim as sim

env_name = f"SDSTests-CustomEnvForReadme-v0"
gym.envs.register(id=env_name, entry_point='social_distancing_sim.environment.gym.environments.sds_746:SDS746',
                  max_episode_steps=1000)
env_spec = gym.make(env_name).spec

sim_ = sim.Sim(env_spec=env_spec,
              agent=VaccinationAgent(actions_per_turn=15))
```

Then create and run MultiSim.

```Python
multi_sim = sim.MultiSim(sim_, name='basic agent comparison',
                         n_reps=300, n_jobs=30)
multi_sim.run()
```

See scripts/stats_compare_basic_agents.py and stats_compare_multi_agents.py for further examples

# Simulation structure and components <a name="package"></a>
The social_distancing_sim package is split into 5 main modules; .sim, .environment, .agent, .gym, and .templates. See docstrings for object parameters and details.

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
    - .gym - Contains environment and agent definitions designed to comply with the [OpenAI Gym API](https://gym.openai.com/). This include the trainable reinforcement learning agents.
      - gym.**gym_env** - Wrapper to make social_distancing_sim.environment.Environments Gym compatible  
      - gym.**gym_templates** - Gym environments specs for example environment set ups in social_distancing_sim.templates  
      - gym.**wrappers** - Various Gym envriroment wrappers   
     
## .agent
Contains the code defining the agent interface and, currently, 4 basic agents.
  - agent_base.**AgentBase** - An abstract class agents should inherit from. Defines interface for action/target selection mechanisms. 
  - dummy_agent.**DummyAgent** - An agent that models the behaviour of many governments by not doing anything.
  - random_agent.**RandomAgent** - An agent that performs a number of random vaccination or isolation actions, with totally random target selection
  - isolation_agent.**IsolationAgent** - An agent that randomly isolates a number of infected + connected nodes and randomly reconnects recovered + isolated nodes.
  - vaccination_agent.**VaccinationAgent** - An agent that randomly vaccinates currently a number of non-infected nodes each turn.
  - masking_agent.**MaskingAgent** - An agent that randomly provides a supply of masks to alive nodes each turn. By default the masks reduce the chance of infection less than immunity, but both the source and target nodes can potentially contribute to the reduction.. 
  - .multi_agents.MultiAgent - Class handling combinations of different agents
  - policy_agents - Combinations of basic agents that act over different time periods.
  - rl_agents - Compatible reinforcement learning algorithms 
  
## .sim
Contains objects to handle running and logging experiments with agent input
  - .sim.**Sim** - Handles the Environment, and an Agent. Steps the simulation, gets actions from agent, passes to env, etc. 
  - .multi_sim.**MultiSim** - Handles running Sim objects multiple times with different seeds. Outputs MLflow logs and aggregated statistics.
 
 ## .templates
Example environment set ups.

# Experiments <a name="Experiments"></a>
The rest of this readme contains a dump of example experiments with outputs, which can be run using the code below or the relevant script in scripts/.

## Social distancing: Worth it? <a name="social_distancing"></a>
![Example cats vs responsible](https://github.com/garethjns/social-distancing-sim/blob/master/images/joined.gif)   
([Discussion](https://new.reddit.com/r/dataisbeautiful/comments/fov56p/oc_comparing_the_effect_of_social_distancing_on/))

The probabilities of inter and and intra community connectivity in the population can be modified to, for example, compare the effects between a densely connected population and a socially distanced population. In this example, fewer connections in the population give the disease fewer opportunities for disease to spread, which slows the progression through the population. This leads to a flatter infection curve and a lower peak burden on the health care system, results in fewer deaths.

````bash
python3 -m scripts.compare_two_populations
````

## Masks: Worth it? <a name="using_masks"></a>
![Example MaskingAgent](https://github.com/garethjns/social-distancing-sim/blob/master/images/masking_agent_example_joined.gif)   
Provision of masks to a population can be done by an agent. Using the default settings, individual masks reduce the chance of infection less than immunity, but both the source and target nodes can wear masks benefiting from a stacked effect (with diminishing returns). Here the DummyAgent does nothing. 

This example is part of a larger example comparing agents/actions - see below for more details. 

````bash
python3 -m scripts.visual_compare_basic_agents_small
````


## Importance of testing: Modifying ObservationSpace test rate <a name="importance_of_testing"></a>
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

## Herd immunity: Comparing immunity effects <a name="herd_immunity"></a>
![Example testing rate](https://github.com/garethjns/social-distancing-sim/blob/master/images/joined_3.gif)

Version 0.2.0 adds incomplete immunity and decay of immunity. These are part of the disease definition, and allow reinfection after a node has survived infection.

````bash
python3 -m scripts.compare_two_diseases_immunity
````  

## Basic agents: Simple action comparison <a name="basic_agents_visual"></a>
![Test basic agents](https://github.com/garethjns/social-distancing-sim/blob/master/images/basic_agent_comparison_score.png)

Runs the full set of basic agents, comparing the effects of their individual actions. Generates visualisations for each (not shown here).

### Run visual examples
````bash
python3 -m scripts.visual_compare_basic_agents
````  
### Run MultiSim and plot histograms
````bash
python3 -m scripts.stats_compare_basic_agents
````  

## Static policies: Combining actions into strategies <a name="static_policies"></a>
![Test multi policy agents](https://github.com/garethjns/social-distancing-sim/blob/master/images/multi_agent_comparison_score.png)

This example runs each policy agent multiple times and aggregates the scores. The policy agents include the following defaults, which can be modified in the script:
 - Treatment agent: Treats up to 5 infected nodes per turn, starting at turn 50.
 - Vaccination agent: Starts vaccinating up to 5 uninfected nodes at turn 60.
 - Distancing agent: Starts social distancing at turn 15, stops soft-disconnecting nodes at turn 55. Starts reconnecting nodes on turn 60. Acts on up to 15 nodes per turn.
 - Masking agent: Gives up to 10 alive nodes a lifetime supply of masks each turn, starting at turn 40.

````bash
python3 -m social-distancing-sim.scripts.stats_compare_multi_agents
````  

## Reinforcement learning agents vs static policies <a name="reinforcement_agents"></a>
![Reinforcement agents](https://github.com/garethjns/social-distancing-sim/blob/master/images/linear_q_learner_2_comparison.png)

Compares two trained reinforcement agents (Linear and deep QAgents) against each other and static policies defined above.

### Train agents
````bash
````  

### Run comparison
````bash
python3 -m scripts.stats_compare_rl_to_baseline
````  
