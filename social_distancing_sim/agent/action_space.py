from dataclasses import dataclass


@dataclass
class ActionSpace:
    """
    Available actions and associated costs.

    TODO: Standardise api and remove **kwargs
    """
    vaccinate_cost: int = -50
    isolate_cost: int = 0

    def vaccinate(self, **kwargs) -> int:
        kwargs["pop"].disease.give_immunity(kwargs["pop"].observation_space.graph.g_.nodes[kwargs["target_node_id"]])
        kwargs["pop"].observation_space.graph.g_.nodes[kwargs["target_node_id"]]["status"].immune = True
        kwargs["pop"].observation_space.graph.g_.nodes[kwargs["target_node_id"]]["last_tested"] = kwargs["step"]

        return self.vaccinate_cost

    def isolate(self, **kwargs) -> int:
        kwargs["pop"].observation_space.graph.isolate_node(kwargs["target_node_id"])
        kwargs["pop"].observation_space.graph.g_.nodes[kwargs["target_node_id"]]["status"].isolated = True

        return self.isolate_cost
