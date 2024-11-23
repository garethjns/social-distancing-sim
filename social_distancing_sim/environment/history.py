from typing import Any, Callable, Dict, List, Optional, SupportsFloat, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from social_distancing_sim.environment.healthcare import Healthcare
from social_distancing_sim.environment.observation_space import ObservationSpace


class History(dict):
    current_clear_key = "Current clear"
    known_current_clear_key = "Known current clear"
    current_infections_key = "Current infections"
    known_current_infections_key = "Known current infections"
    immune_key = "immune"
    total_immune_key = "Total immune"
    known_total_immune_key = "Known total immune"
    mean_immunity_immune_key = "Mean immunity (of immune nodes)"
    mean_immunity_alive_key = "Mean immunity (of all alive nodes)"
    known_mean_immunity_immune_key = "Known mean immunity (of immune nodes)"
    known_mean_immunity_alive_key = "Known mean immunity (of all alive nodes)"
    total_deaths_key = "Total deaths"
    current_infection_prop_key = "Current infection prop"
    known_current_infection_prop_key = "Known current infection prop"
    current_death_prop_key = "Current death prop"
    overall_death_prop_key = "Overall death prop"
    overall_infected_death_rate_key = "Overall Infected death rate"
    known_overall_infected_death_rate_key = "Known overall Infected death rate"
    total_masked_key = "Total masked"
    known_masked_key = "Known masked"
    total_recovered_key = "Total recovered"
    total_infections_key = "Total infections"
    known_total_infections_key = "Known total infections"
    overall_score_key = "Overall score"
    observed_overall_score_key = "Observed overall score"
    overall_infection_prop_key = "Overall infection prop"
    known_overall_infection_prop_key = "Known overall infection prop"
    turn_score_key = "Turn score"
    observed_turn_score_key = "Observed turn score"
    new_infections_key = "New infections"
    known_new_infections_key = "Known new infections"
    new_deaths_key = "New deaths"
    current_recoveries_key = "Current recoveries"
    current_infection_rate_penalty_key = "Current recovery rate penalty"
    number_alive_key = "Number alive"

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
        hist.colours = {
            cls.current_clear_key: "#1f77b4",
            cls.known_current_clear_key: "#1f77b4",
            cls.current_infections_key: "#d62728",
            cls.known_current_infections_key: "#d62728",
            cls.total_immune_key: "#9467bd",
            cls.known_total_immune_key: "#9467bd",
            cls.mean_immunity_immune_key: "#6e4196",
            cls.mean_immunity_alive_key: "#9467bd",
            cls.known_mean_immunity_immune_key: "#6e4196",
            cls.known_mean_immunity_alive_key: "#9467bd",
            cls.total_deaths_key: "k",
            cls.current_infection_prop_key: "#1f77b4",
            cls.known_current_infection_prop_key: "#1f77b4",
            cls.current_death_prop_key: "k",
            cls.overall_infected_death_rate_key: "k",
            cls.known_overall_infected_death_rate_key: "k",
        }

        return hist

    @property
    def last_turn(self) -> Dict[str, float]:
        """Return last logged value for each key."""
        return {k: v[-1] if len(v) > 0 else v for k, v in self.items()}

    def plot(
        self,
        ks: List[str],
        ax: plt.axes = None,
        x_label: str = "Day",
        y_label: str = "Count",
        show: bool = True,
        x_lim: Tuple[SupportsFloat, SupportsFloat] = None,
        y_lim: Tuple[SupportsFloat, SupportsFloat] = None,
        remove_x_tick_labels: bool = False,
        agg_f: Callable = None,
    ) -> plt.axes:
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
            _, ax = plt.subplots(nrows=1, ncols=1)

        if agg_f is None:

            def agg_f(x):
                return x

        for k in ks:
            y = agg_f(self[k])
            ax.plot(y, label=k, color=self.colours.get(k, None))

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

    def log_score(
        self,
        recoveries: int,
        known_new_infections: int,
        new_infections: int,
        deaths: int,
        turn_score: float = 0.0,
        obs_turn_score: float = 0.0,
    ):
        # Log things that affect score
        self.log(
            {
                self.turn_score_key: turn_score,
                self.observed_turn_score_key: obs_turn_score,
                self.new_infections_key: new_infections,
                self.known_new_infections_key: known_new_infections,
                self.new_deaths_key: deaths,
                self.current_recoveries_key: recoveries,
            }
        )

    def log_actions(
        self,
        actions_attempted: Dict[int, Optional[int]],
        actions_taken: Dict[int, int],
    ):
        # Log actions
        for suffix, actions_dict in zip(
            ("attempted", "completed"), (actions_attempted, actions_taken)
        ):
            self.log(
                {
                    f"Actions {suffix}": len(actions_dict.values()),
                    f"Vaccinate actions {suffix}": len(
                        [a for a in actions_dict.values() if a == 1]
                    ),
                    f"Isolate actions {suffix}": len(
                        [a for a in actions_dict.values() if a == 2]
                    ),
                    f"Reconnect actions {suffix}": len(
                        [a for a in actions_dict.values() if a == 3]
                    ),
                    f"Treat actions {suffix}": len(
                        [a for a in actions_dict.values() if a == 4]
                    ),
                    f"Mask actions {suffix}": len(
                        [a for a in actions_dict.values() if a == 5]
                    ),
                }
            )

    def log_observation_space(self, obs: ObservationSpace, healthcare: Healthcare):
        # Log full space and observed space
        total_deaths = len(obs.graph.current_dead_nodes)
        total_infections = np.sum(self[self.new_infections_key])
        total_population = obs.graph.total_population
        self.log(
            {
                self.current_infections_key: obs.graph.n_current_infected,
                self.known_current_infections_key: obs.known_n_current_infected,
                self.current_clear_key: total_population - obs.graph.n_current_infected,
                self.known_current_clear_key: (
                    total_population - obs.known_n_current_infected
                ),
                self.current_infection_rate_penalty_key: healthcare.recovery_rate_penalty(
                    obs.graph.n_current_infected
                ),
                self.number_alive_key: len(obs.graph.current_alive_nodes),
                self.total_deaths_key: total_deaths,
                self.total_immune_key: len(obs.graph.current_immune_nodes),
                self.mean_immunity_immune_key: np.mean(
                    [
                        obs.graph.g_.nodes[n][self.immune_key]
                        for n in obs.graph.current_immune_nodes
                    ]
                ),
                self.mean_immunity_alive_key: np.mean(
                    [
                        obs.graph.g_.nodes[n].get(self.immune_key, 0)
                        for n in obs.graph.current_alive_nodes
                    ]
                ),
                self.known_total_immune_key: len(obs.current_immune_nodes),
                self.known_mean_immunity_immune_key: np.mean(
                    [
                        obs.graph.g_.nodes[n][self.immune_key]
                        for n in obs.current_immune_nodes
                    ]
                ),
                self.known_mean_immunity_alive_key: np.mean(
                    [
                        obs.graph.g_.nodes[n].get(self.immune_key, 0)
                        for n in obs.current_alive_nodes
                    ]
                ),
                self.total_masked_key: len(obs.graph.current_masked_nodes),
                self.known_masked_key: len(obs.current_masked_nodes),
                self.total_recovered_key: np.sum(self[self.current_recoveries_key]),
                self.total_infections_key: total_infections,
                self.known_total_infections_key: np.sum(
                    self[self.known_new_infections_key]
                ),
                self.overall_score_key: np.sum(self[self.turn_score_key]),
                self.observed_overall_score_key: np.sum(
                    self[self.observed_turn_score_key]
                ),
            }
        )

        # Props/rates
        self.log(
            {
                self.current_infection_prop_key: self[self.current_infections_key][-1]
                / total_population,
                self.known_current_infection_prop_key: (
                    self[self.known_current_infections_key][-1] / total_population
                ),
                self.overall_infection_prop_key: total_infections / total_population,
                self.known_overall_infection_prop_key: (
                    self[self.known_total_infections_key][-1] / total_population
                ),
                self.current_death_prop_key: total_deaths / total_population,
                self.overall_death_prop_key: total_deaths / total_population,
                self.overall_infected_death_rate_key: (total_deaths / total_infections),
                self.known_overall_infected_death_rate_key: total_deaths
                / total_infections,
            }
        )
