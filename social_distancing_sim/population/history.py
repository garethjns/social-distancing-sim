from typing import Dict, Callable, Tuple, List, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator


class History(dict):
    def __init__(self, *args, colours: Dict[str, str] = None):
        super().__init__(*args)
        if colours is None:
            colours = {}
        self.colours = colours

    def __missing__(self, k):
        self[k] = []
        return self[k]

    @classmethod
    def with_fields(cls, fields: List[str]) -> "History":
        return History({k: [] for k in fields})

    @classmethod
    def with_defaults(cls) -> "History":
        """Set fields or colours for use with sim."""

        default_fields = ["Current recovery rate", "Current infections", "Current clear", "Total immune",
                          "Number alive", "New infections", "New deaths", "Total deaths", "Total recovered",
                          "Current Infection prop", "Overall infection prop", "Current death prop",
                          "Overall death prop", "graph", "Overall Infected death rate"]

        hist = History.with_fields(default_fields)

        hist.colours = {'Current clear': '#1f77b4',
                        'Current infections': '#d62728',
                        'Total immune': '#9467bd',
                        'Total deaths': 'k',
                        'Current infection prop': '#1f77b4',
                        'Current death prop': 'k',
                        'Overall Infected death rate': 'k'}

        return hist

    def plot(self, ks: List[str],
             ax: plt.axes = None,
             y_label: str = 'Count',
             x_label: str = 'Day',
             show: bool = True,
             agg_f: Callable = None) -> plt.axes:
        """
        Plot history items.

        :param ks: List of keys to plot.
        :param ax: Axis to draw figure on. Not currently will rewrite axis labels. If not specified, will create.
        :param y_label: Y axis label.
        :param x_label: X axis label.
        :param show: Draw figure.
        :param agg_f: Aggregator function apply to y before plotting, eg. np.cumsum for cumulative plots.
                      Default no aggregation.
        :return: Axis handle used.
        """
        if ax is None:
            fig, ax = plt.subplots(nrows=1,
                                   ncols=1)

        if agg_f is None:
            agg_f = lambda x: x

        for k in ks:
            y = agg_f(self[k])
            ax.plot(y, label=k,
                    color=self.colours.get(k, None))

        # ax.set_ylim([-10, 600])
        ax.set_ylabel(y_label)
        ax.set_xlabel(x_label)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.legend()
        if show:
            plt.show()

        return ax
