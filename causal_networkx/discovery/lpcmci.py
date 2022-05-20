from causal_networkx.discovery import ConstraintDiscovery


class LPCMCI(ConstraintDiscovery):
    def __init__(
        self,
        ci_estimator: Callable,
        alpha: float = 0.05,
        init_graph: Union[nx.Graph, CausalGraph] = None,
        fixed_edges: nx.Graph = None,
        max_cond_set_size: int = None,
        **ci_estimator_kwargs
    ):
        super().__init__(
            ci_estimator, alpha, init_graph, fixed_edges, max_cond_set_size, **ci_estimator_kwargs
        )

        # functions to apply rule orientations
        # during the ancestral removal phase
        self._rule_order_prelim = [
            self._apply_apr,
            self._apply_mmr,
            self._apply_erule8,
            self._apply_erule2,
            self._apply_erule9,
            self._apply_erule10,
        ]

        self._rule_order_prelim = [
            self._apply_apr,
            self._apply_mmr,
            self._apply_erule8,
            self._apply_erule2,
            self._apply_erule1,
            self._apply_erule0d,
            self._apply_erule0c,
            self._apply_erule3,
            self._apply_rule4,
            self._apply_erule9,
            self._apply_erule10,
            self._apply_erule0b,
            self._apply_erule0a,
        ]

    def _apply_apr(self):
        pass

    def _apply_mmr(self):
        pass
