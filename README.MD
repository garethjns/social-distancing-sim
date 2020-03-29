# Social distancing simulator
![Testing is key](https://github.com/garethjns/social-distancing-sim/workflows/Testing%20is%20key/badge.svg)

Social distancing is an unfortunately unclear term; it means stay from other people to avoid killing yourself and them.

But why?

This package models disease spread through a population consisting of sub communities. These communities are inter and intra connected, the probability of these connections can be modified to model "normal" populations and populations exercising social distancing. Fewer connections mean fewer opportunities for disease to spread, slower progression through the population, a lower peak burden on the health care system, and fewer deaths..   

The code aims to be as simple and understandable as possible, and can be summarised in two graphs: 

![Example cats vs responsible](https://github.com/garethjns/social-distancing-sim/blob/master/images/joined.gif) 

## Population dynamics

The dynamics of this simulation aim to be simple but interesting. They're built from reasoning and guesses at sensible values. They're not trained from data. The examples and default values aim to create simulations that do things in reasonable time, feel free to tweak them and see what happens.

Populations are randomly generated using a [networkx.random_partition_graph](https://networkx.github.io/documentation/stable/reference/generated/networkx.generators.community.random_partition_graph.html#networkx.generators.community.random_partition_graph). This creates a network consisting of communities where individual members have a given chance to be connected. Each individual member also has a lower chance to be connected to members of other communities.

In addition to communities, populations define a healthcare capacity. When above this capacity, the recovery rate from the disease is reduced.

The connections between individuals (graph nodes) define opportunities for a member to infect another. Each day (step) every infected node has one chance to infect each of it's neighbours, the chance of this happening is defined by the disease virulence.

Each day, infected nodes also have the chance to end their infection. The chance of this happening grows with the length of time the individual has been infected. If the infection ends, the individual either recovers and gains immunity, or dies. The chance of recovery is defined by the recovery rate of the disease, modified by the current burden on the healthcare system. When the heathcare system is below capacity, no penalty is applied to the recovery rate. When it's above, the recovery rate is reduced proportionally to the size of the burden. 


# Running in Docker
```bash
docker pull garethjns/social-distancing-sim
docker run -p 8000:8000 social-distancing-sim
```

For now, REST API supports get requests for running a few examples. These return either the rendered gif, or the stats for the simulation (JSON). See http://localhost:8001/docs for all implemented methods.

For example (these will probably take a while to run):  
http://localhost:8000/examples/herd_of_cats -> outputs gif  
http://localhost:8000/examples/herd_of_cats_multiple_runs -> outputs json (if running through a browser, Firefox does a better job of displaying this than Chrome).  

# Running in Python
````bash
pip install social_distancing_sim
````


## Run a simulation
See also scripts/run_single_population.py

````python
from social_distancing_sim.disease.disease import Disease
from social_distancing_sim.population.population import Population

# Define a disease
disease = Disease(name='COVID-19')

# Create a population
pop = Population(name="example population",
                 disease=disease,
                 community_n=50,  # Numebr of communities
                 community_size_mean=15,  # Average number of people per community
                 community_p_in=0.06,  # The likihood of a connection between two community members
                 community_p_out=0.04)  # The liklikhood of a connection between two members of different communities.  

# Run the simulation for a number of days
pop.run(steps=100)

# Save .gif to './example population/replay.gif'
pop.replay()

# History can be accessed in the history attribute
pop.history.keys()
pop.history(list(pop.history.keys()))
````

Or run with
````bash
python3 -m social-distancing-sim.scripts.run_single_population
````


## Compare two populations

```python
from social_distancing_sim.disease.disease import Disease
from social_distancing_sim.population.population import Population

disease = Disease(name='COVID-19')

pop = Population(name='A herd of cats',
                 disease=disease,
                 community_n=50,
                 community_size_mean=15,
                 seed=123)

pop_distanced = Population(name='A socially responsible population',
                           disease=disease,
                           community_n=50,
                           community_size_mean=15,
                           community_p_in=0.05,
                           community_p_out=0.04,
                           seed=123)

pop_distanced.run(steps=200,
                  plot=False)
pop.run(steps=200,
        plot=False)

pop_distanced.replay(duration=0.1)
pop.replay(duration=0.1)
```

Or run with
````bash
python3 -m social-distancing-sim.scripts.compare_two_populations
````


