from typing import List

import matplotlib.pyplot as plt
import numpy as np


class History(dict):
    @classmethod
    def with_defaults(cls) -> "History":
        return History({"Current recovery rate": [],
                        "Current infections": [],
                        "Number alive": [],
                        "New infections": [],
                        "New deaths": [],
                        "Total deaths": [],
                        "Total recovered": [],
                        "graph": []})

    def plot(self, ks: List[str],
             ax: plt.axes = None,
             y_label: str = 'Count',
             x_label: str = 'Day',
             show: bool = True) -> plt.axes:
        if ax is None:
            fig, ax = plt.subplots(nrows=1,
                                   ncols=1)

        for k in ks:
            ax.plot(self[k], label=k)

        ax.set_ylabel(y_label)
        ax.set_xlabel(x_label)
        plt.legend()
        if show:
            plt.show()

        return ax

    def plot_cumulative(self, ks: List[str],
                        ax: plt.axes = None,
                        y_label: str = 'Count',
                        show: bool = True) -> plt.axes:
        if ax is None:
            fig, ax = plt.subplots(nrows=2,
                                   ncols=1)

        for k in ks:
            ax[0].plot(self[k], label=k)
            ax[1].plot([0] + list(np.cumsum(self[k])), label=k)

        ax[0].set_ylabel(y_label)
        ax[1].set_ylabel(y_label)
        ax[1].set_xlabel('Step')
        plt.legend()

        if show:
            plt.show()

        return ax
