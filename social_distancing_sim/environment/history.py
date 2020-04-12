from typing import Dict, Callable, List, Any, Tuple, SupportsFloat

import matplotlib.pyplot as plt
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
        hist = History()
        hist.colours = {'Current clear': '#1f77b4',
                        'Known current clear': '#1f77b4',
                        'Current infections': '#d62728',
                        'Known current infections': '#d62728',
                        'Total immune': '#9467bd',
                        'Known total immune': '#9467bd',
                        'Mean immunity (of immune nodes)': '#6e4196',
                        'Mean immunity (of all alive nodes)': '#9467bd',
                        'Known mean immunity (of immune nodes)': '#6e4196',
                        'Known mean immunity (of all alive nodes)': '#9467bd',
                        'Total deaths': 'k',
                        'Current infection prop': '#1f77b4',
                        'Known current infection prop': '#1f77b4',
                        'Current death prop': 'k',
                        'Overall Infected death rate': 'k',
                        'Known overall Infected death rate': 'k'}

        return hist

    def plot(self, ks: List[str],
             ax: plt.axes = None,
             x_label: str = 'Day',
             y_label: str = 'Count',
             show: bool = True,
             x_lim: Tuple[SupportsFloat, SupportsFloat] = None,
             y_lim: Tuple[SupportsFloat, SupportsFloat] = None,
             remove_x_tick_labels: bool = False,
             agg_f: Callable = None) -> plt.axes:
        """
        Plot history items.

        :param ks: List of keys to plot.
        :param ax: Axis to draw figure on. Not currently will rewrite axis labels. If not specified, will create.
        :param x_label: X axis label.
        :param y_label: Y axis label.
        :param x_lim:
        :param y_lim:
        :param remove_x_tick_labels:
        :param show: Draw figure.
        :param agg_f: Aggregator function apply to y before plotting, eg. np.cumsum for cumulative plots.
                      Default no aggregation.
        :return: Axis handle used.
        """
        if ax is None:
            fig, ax = plt.subplots(nrows=1,
                                   ncols=1)

        if agg_f is None:
            def agg_f(x):
                return x

        for k in ks:
            y = agg_f(self[k])
            ax.plot(y, label=k,
                    color=self.colours.get(k, None))

        if y_lim is not None:
            ax.set_ylim(y_lim)
        if x_lim is not None:
            ax.set_xlim(x_lim)
        if remove_x_tick_labels:
            ax.set_xticklabels([])
        ax.set_ylabel(y_label)
        ax.set_xlabel(x_label)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend()
        if show:
            plt.show()

        return ax

    def log(self, metrics: Dict[str, Any]):
        for k, v in metrics.items():
            self[k].append(v)
