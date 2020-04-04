class ActionSpace:
    @staticmethod
    def vaccinate(**kwargs) -> None:
        kwargs["disease"].give_immunity(kwargs["pop"].observation_space.graph.g_.nodes[kwargs["target_node_id"]])

    @staticmethod
    def isolate(**kwargs) -> None:
        kwargs["pop"].observation_space.graph.isolate_node(kwargs["target_node_id"])
