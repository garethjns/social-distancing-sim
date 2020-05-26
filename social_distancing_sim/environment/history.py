from typing import Dict, Callable, List, Any, Tuple, SupportsFloat

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from social_distancing_sim.environment.healthcare import Healthcare
from social_distancing_sim.environment.observation_space import ObservationSpace


class History(dict):
    def __init__(self, *args, colours: Dict[str, str] = None) -> None:
        super().__init__(*args)
        if colours is None:
            colours = {}
        self.colours = colours

    def __missing__(self, k) -> List[Any]:
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

    def log(self, metrics: Dict[str, Any]) -> None:
        for k, v in metrics.items():
            self[k].append(v)

    def log_score(self, recoveries: int, known_new_infections: int, new_infections: int, deaths: int,
                  turn_score: float = 0.0, obs_turn_score: float = 0.0):
        # Log things that affect score
        self.log({"Turn score": turn_score, "Observed turn score": obs_turn_score,
                  "New infections": new_infections, "Known new infections": known_new_infections,
                  "New deaths": deaths, "Current recovered": recoveries})

    def log_actions(self, actions_attempted: Dict[int, int], actions_taken: Dict[int, int], ):
        # Log actions
        for suffix, actions_dict in zip(('attempted', 'completed'), (actions_attempted, actions_taken)):
            self.log({f"Actions {suffix}": len(actions_dict.values()),
                      f"Vaccinate actions {suffix}": len([a for a in actions_dict.values() if a == 1]),
                      f"Isolate actions {suffix}": len([a for a in actions_dict.values() if a == 2]),
                      f"Reconnect actions {suffix}": len([a for a in actions_dict.values() if a == 3]),
                      f"Treat actions {suffix}": len([a for a in actions_dict.values() if a == 4]),
                      f"Mask actions {suffix}": len([a for a in actions_dict.values() if a == 5])})

    def log_observation_space(self, obs: ObservationSpace, healthcare: Healthcare):
        # Log full space and observed space
        total_deaths = len(obs.graph.current_dead_nodes)
        total_infections = np.sum(self["New infections"])
        total_population = obs.graph.total_population
        self.log({"Current infections": obs.graph.n_current_infected,
                  "Known current infections": obs.known_n_current_infected,
                  "Current clear": total_population - obs.graph.n_current_infected,
                  "Known current clear": (total_population - obs.known_n_current_infected),
                  "Current recovery rate penalty": healthcare.recovery_rate_penalty(obs.graph.n_current_infected),
                  "Number alive": len(obs.graph.current_alive_nodes),
                  "Total deaths": total_deaths,
                  "Total immune": len(obs.graph.current_immune_nodes),
                  "Mean immunity (of immune nodes)": np.mean([obs.graph.g_.nodes[n]["immune"]
                                                              for n in obs.graph.current_immune_nodes]),
                  "Mean immunity (of all alive nodes)": np.mean([obs.graph.g_.nodes[n].get("immune", 0)
                                                                 for n in obs.graph.current_alive_nodes]),
                  "Known total immune": len(obs.current_immune_nodes),
                  "Known mean immunity (of immune nodes)": np.mean([obs.graph.g_.nodes[n]["immune"]
                                                                    for n in obs.current_immune_nodes]),
                  "Known mean immunity (of all alive nodes)": np.mean([obs.graph.g_.nodes[n].get("immune", 0)
                                                                       for n in obs.current_alive_nodes]),
                  "Total masked": len(obs.graph.current_masked_nodes),
                  "Known masked": len(obs.current_masked_nodes),
                  "Total recovered": np.sum(self["Current recoveries"]),
                  "Total infections": total_infections,
                  "Known total infections": np.sum(self["Known new infections"]),
                  "Overall score": np.sum(self["Turn score"]),
                  "Observed overall score": np.sum(self["Observed turn score"])})

        # Props/rates
        self.log({"Current infection prop": self["Current infections"][-1] / total_population,
                  "Known current infection prop": (self["Known current infections"][-1] / total_population),
                  "Overall infection prop": total_infections / total_population,
                  "Known overall infection prop": (self["Known total infections"][-1] / total_population),
                  "Current death prop": total_deaths / total_population,
                  "Overall death prop": total_deaths / total_population,
                  "Overall Infected death rate": (total_deaths / total_infections),
                  "Known overall Infected death rate": total_deaths / total_infections})
