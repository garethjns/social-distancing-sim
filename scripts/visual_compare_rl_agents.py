"""
Run a single simulation to see what the RL agents (linear q learner so far) do.

This agent appears to learn to spam isolation actions. It hasn't learnt to reconnect nodes. It has no pressure to stop
spamming isolation actions as there is no cost, and they're not performed anyway due to the environment logic to target
infected nodes only.
"""

import social_distancing_sim.environment as env
from social_distancing_sim.gym.agent.rl.q_learners.linear_q_agent import LinearQAgent
from social_distancing_sim.sim.sim import Sim
from social_distancing_sim.templates.small import Small

if __name__ == "__main__":
    steps = 200
    linear_q_agent = LinearQAgent.load('linear_q_learner.pkl')
    linear_q_agent.name = 'linear_q_agent'
    linear_q_agent.actions_per_turn = 1

    pop = Small().build()
    pop.name = 'Linear q agent in Small template'
    pop.environment_plotting = env.EnvironmentPlotting(ts_fields_g2=["Vaccinate actions", "Isolate actions",
                                                                     "Reconnect actions", "Treat actions"])
    pop.environment_plotting.prepare_output_path(name=pop.name)

    sim = Sim(env=pop,
              agent=linear_q_agent,
              n_steps=steps,
              tqdm_on=True,
              plot=True,
              save=True)

    sim.run()
    print(sim.env.history["Actions taken"])
    print(sim.env.history["Isolate actions attempted"])
    print(sim.env.history["Isolate actions completed"])
