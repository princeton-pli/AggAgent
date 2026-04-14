"""Aggregation strategies."""

from .base import Strategy, MetricResult, CostBreakdown
from .pass_strategy import PassStrategy
from .mv import MVStrategy, WMVStrategy
from .bon import BONStrategy
from .fewtool import FewToolStrategy
from .llm_based import SolAgg, SummAgg
from .aggagent import AggAgent

__all__ = [
    "Strategy",
    "MetricResult",
    "CostBreakdown",
    "PassStrategy",
    "MVStrategy",
    "WMVStrategy",
    "BONStrategy",
    "FewToolStrategy",
    "SolAgg",
    "SummAgg",
    "AggAgent",
    "STRATEGIES",
    "HEURISTIC_STRATEGIES",
    "get_strategy",
    "get_heuristic_strategies",
]

# All strategies (heuristic + LLM-based)
STRATEGIES = {
    "pass": PassStrategy,
    "mv": MVStrategy,
    "wmv": WMVStrategy,
    "bon": BONStrategy,
    "fewtool": FewToolStrategy,
    "solagg": SolAgg,
    "summagg": SummAgg,
    "aggagent": AggAgent,
}

# Heuristic-only (no LLM calls)
HEURISTIC_STRATEGIES = {
    "pass": PassStrategy,
    "mv": MVStrategy,
    "wmv": WMVStrategy,
    "bon": BONStrategy,
    "fewtool": FewToolStrategy,
}


def get_strategy(name: str, **kwargs) -> Strategy:
    """
    Get a strategy instance by name.

    Args:
        name: Strategy name (pass, mv, wmv, bon, fewtool, sol, summ)
        **kwargs: Forwarded to the strategy constructor; unknown keys are absorbed via **kwargs
    """
    if name not in STRATEGIES:
        raise ValueError(f"Unknown strategy: {name}. Available: {list(STRATEGIES.keys())}")
    return STRATEGIES[name](**kwargs)


def get_heuristic_strategies(**kwargs) -> list[Strategy]:
    """
    Get instances of all heuristic (non-LLM) strategies.

    Args:
        **kwargs: Forwarded to each constructor; unknown keys are absorbed via **kwargs
    """
    return [cls(**kwargs) for cls in HEURISTIC_STRATEGIES.values()]
