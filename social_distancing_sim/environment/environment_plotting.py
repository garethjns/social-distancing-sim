import copy
import glob
import os
import shutil
from dataclasses import dataclass
from typing import List, Union

import imageio
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from social_distancing_sim.environment.healthcare import Healthcare
from social_distancing_sim.environment.history import History
from social_distancing_sim.environment.observation_space import ObservationSpace


@dataclass
class EnvironmentPlotting:
    both: bool = True
    ts_fields_g1: List[str] = None
    ts_fields_g2: List[str] = None
    ts_obs_fields_g1: List[str] = None
    ts_obs_fields_g2: List[str] = None

    def __post_init__(self):
        self.name: Union[str, None] = None
        self.output_path: Union[str, None] = None
        self.graph_path: Union[str, None] = None
        self.name: Union[str, None] = None

        sns.set()

    def prepare_output_path(self, name: str) -> None:
        if self.output_path is None:
            self.name = name
            self.output_path = f"{name}"
            shutil.rmtree(self.output_path,
                          ignore_errors=True)

            self.graph_path = f"{self.output_path}/graphs/"
            os.makedirs(self.graph_path,
                        exist_ok=True)

    def _prepare_figure(self, test_rate: float = 1) -> None:
        """
        Prepare the main output figure

        This has:
        - 2x row for network plot                   |  2x row for network plot (if testing rate < 1)
        - 1x row for ts plot                        | 1x row for ts plot (if testing rate < 1)
        - 1x row for additional ts plot (optional)  | 1x row for additional ts plot (optional, if testing rate < 1)
        """
        plt.close()

        self._g2_on = False
        ts_ax_g2 = None
        self.test_rate = test_rate

        if self.ts_fields_g1 is None:
            self.ts_fields_g1 = ["Current infections", "Total immune", "Total deaths"]
        if self.ts_obs_fields_g1 is None:
            self.ts_obs_fields_g1 = ["Known current infections", "Known total immune", "Total deaths"]
        if self.ts_fields_g2 is None:
            self.ts_fields_g2 = []
        if self.ts_obs_fields_g2 is None:
            self.ts_obs_fields_g2 = []

        if len(self.ts_fields_g2) > 0:
            self._g2_on = True

        height = 5
        nrows = 6
        if self._g2_on:
            height += height / nrows * 2
            nrows += 2

        if (self.test_rate < 1) & self.both:
            # Plot reality and observed space separately
            fig = plt.figure(figsize=(height * 2, height))
            gs = fig.add_gridspec(nrows, 2)
            graph_ax = [fig.add_subplot(gs[:4, 0]), fig.add_subplot(gs[:4, 1])]
            ts_ax_g1 = [fig.add_subplot(gs[4:6, 0]), fig.add_subplot(gs[4:6, 1])]
            if self._g2_on:
                ts_ax_g2 = [fig.add_subplot(gs[6:8, 0]), fig.add_subplot(gs[6:8, 1])]
        else:
            # Observed is reality, just plot single figure
            fig = plt.figure(figsize=(6.4, height))
            gs = fig.add_gridspec(nrows, 1)
            graph_ax = [fig.add_subplot(gs[:4, 0])]
            ts_ax_g1 = [fig.add_subplot(gs[4:6, 0])]
            if self._g2_on:
                ts_ax_g2 = [fig.add_subplot(gs[6:8, 0])]

        self._figure = fig
        self._graph_ax: List[plt.Axes] = graph_ax
        self._ts_ax_g1: List[plt.Axes] = ts_ax_g1
        self._ts_ax_g2: List[plt.Axes] = ts_ax_g2

    def plot(self, obs: ObservationSpace, history: History, healthcare: Healthcare, step: int,
             total_steps: int,
             save: bool = True, show: bool = True) -> None:
        self._prepare_figure(test_rate=obs.test_rate)
        self.plot_graphs(obs=obs, title=f"{self.name}, day {step} (deaths = {len(obs.graph.current_dead_nodes)})")
        self.plot_ts(history=history, healthcare=healthcare, step=step,
                     total_steps=total_steps, total_population=obs.graph.total_population)

        self._figure.tight_layout()

        if save:
            plt.savefig(f"{self.output_path}/graphs/{step}_graph.png")

        if show:
            plt.show()

    def plot_ts(self, history: History, healthcare: Healthcare, total_steps: int, total_population: int,
                step: int) -> None:
        for ax, fields in zip(self._ts_ax_g1, [self.ts_fields_g1, self.ts_obs_fields_g1]):
            history.plot(ks=fields,
                         x_lim=(-1, total_steps),
                         y_lim=(-10, int(total_population + total_population * 0.05)),
                         x_label='Day' if not self._g2_on else None,
                         remove_x_tick_labels=self._g2_on,
                         ax=ax,
                         show=False)
            ax.plot([0, step], [healthcare.capacity, healthcare.capacity],
                    linestyle="--",
                    color='k')

        if self._g2_on:
            for ax, fields in zip(self._ts_ax_g2, [self.ts_fields_g2, self.ts_obs_fields_g2]):
                history.plot(ks=fields,
                             y_label='Score',
                             x_lim=(-1, total_steps),
                             # y_lim=(-0.1, 1.1),
                             ax=ax,
                             show=False)

    def plot_graphs(self, obs: ObservationSpace, title: str):
        obs.graph.plot(ax=self._graph_ax[0])
        self._graph_ax[0].set_title(f"Full sim: {title}")

        if (obs.test_rate < 1) & self.both:
            obs.plot(ax=self._graph_ax[1])
            self._graph_ax[1].set_title(f"Observed: {title}")

    def replay(self, duration: float = 0.2) -> str:
        """
        :param duration: Frame duration,
        :return: Path to rendered gif.
        """
        # Find all previously saved steps
        fns = glob.glob(f"{self.graph_path}*_graph.png")
        # Ensure ordering
        fns = [f.replace('\\', '/') for f in fns]
        sorted_idx = np.argsort([int(f.split('_graph.png')[0].split(self.graph_path)[1]) for f in fns])
        fns = np.array(fns)[sorted_idx]

        # Generate gif
        output_path = f"{self.output_path}/replay.gif"
        images = [imageio.imread(f) for f in fns]
        imageio.mimsave(output_path, images,
                        duration=duration,
                        subrectangles=True)

        return output_path

    def clone(self) -> "EnvironmentPlotting":
        return copy.deepcopy(self)
