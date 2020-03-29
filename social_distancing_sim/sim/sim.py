import warnings
from dataclasses import dataclass
from typing import List

import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from social_distancing_sim.population.population import Population


@dataclass
class Sim:
    populations: List[Population]
    mode: str = 'visual'
    steps: int = 100
    reps: int = 1
    n_jobs: int = -2

    def __post_init__(self) -> None:
        expanded_p = []
        for p in self.populations:
            expanded_p.extend([p] * self.reps)
        self._populations = expanded_p

        if (self.mode.lower() == 'visual') & (self.reps > 1):
            warnings.warn('Visual mode but reps > 1, setting to 1.')
            self.reps = 1

        self.outputs: List[str] = []
        self.results: pd.DataFrame = pd.DataFrame()

    async def run(self) -> None:
        res = Parallel(n_jobs=self.n_jobs,
                       backend='loky')(delayed(self._run)(p) for p in tqdm(self._populations))

        fvs = []
        for ri, r in enumerate(res):
            fvs.append(pd.DataFrame({k: v[-1] for k, v in r.history.items() if k not in ["graph"]},
                                    index=[r.name]))

        self.results = pd.concat(fvs,
                                       axis=0).reset_index(drop=False).rename({'index': 'name'},
                                                                              axis=1)

    def _run(self, p: Population) -> Population:
        p_ = p.reset()

        if self.mode == 'visual':
            p_.run(steps=self.steps,
                   plot=False,
                   save=True)
            p.replay(duration=0.5)
        else:
            p_.seed = None
            p_.run(steps=self.steps,
                   plot=False,
                   save=False)

        return p_

    def summary(self) -> pd.DataFrame:
        means = self.results.groupby('name').mean()
        means.columns = [f"{c}_mean" for c in means]
        stds = self.results.groupby('name').std()
        stds.columns = [f"{c}_std" for c in stds]

        return pd.concat((means, stds),
                         axis=1)


if __name__ == "__main__":
    from social_distancing_sim.population.population_templates import herd_of_cats, socially_responsible

    import asyncio

    async def run_sim(sim):
        await sim.run()
        return sim

    loop = asyncio.get_event_loop()

    ss = Sim(populations=[herd_of_cats, socially_responsible],
             mode='stats',
             reps=5)

    ss = loop.run_until_complete(run_sim(ss))

    vs = Sim(populations=[herd_of_cats],
             reps=1)

    loop.run_until_complete(run_sim(vs))

