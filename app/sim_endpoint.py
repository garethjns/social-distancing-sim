import json

import gym
import numpy as np
from fastapi import FastAPI
from starlette.responses import FileResponse

from social_distancing_sim.environment import History
from social_distancing_sim.sim import Sim
from social_distancing_sim.templates.gym.register import register_template_envs

app = FastAPI()

register_template_envs()


class SimEndpoint:

    @staticmethod
    @app.get('/ping')
    async def ping():
        return {'response': 'hi'}

    @staticmethod
    @app.get('/run_visual')
    async def run_visual(env: str = 'SDS-small-v0', steps: int = 40):
        sim = Sim(env_spec=gym.make(env).spec, n_steps=steps, plot=False, save=True, tqdm_on=True)
        sim.run()

        return FileResponse(f"{sim.agent.env.sds_env.environment_plotting.output_path}/replay.gif")

    @staticmethod
    @app.get('/run')
    async def run(env: str = 'SDS-small-v0', steps: int = 300):
        sim = Sim(env_spec=gym.make(env).spec, n_steps=steps, plot=False, save=False, tqdm_on=True)
        hist = sim.run()

        return json.dumps(_json_safe(hist))


def _json_safe(hist: History):
    new_hist = {}
    for k, v in hist.items():
        if isinstance(v, list) & (len(v) > 0):
            if isinstance(v[0], np.integer):
                v = [int(vi) for vi in v]
            if isinstance(v[0], np.floating):
                v = [float(vi) for vi in v]

            new_hist[k] = v

    return new_hist
