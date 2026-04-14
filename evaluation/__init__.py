from __future__ import annotations

from .browsecomp import BrowseCompEvaluator, BROWSECOMP_INSTRUCTION
from .deepsearchqa import DeepSearchQAEvaluator, DEEPSEARCHQA_INSTRUCTION
from .healthbench import HealthBenchEvaluator
from .researchrubrics import ResearchRubricsEvaluator


_BROWSECOMP_EVAL = BrowseCompEvaluator()
_DEEPSEARCHQA_EVAL = DeepSearchQAEvaluator()
_HEALTHBENCH_EVAL = HealthBenchEvaluator()
_RESEARCHRUBRICS_EVAL = ResearchRubricsEvaluator()


def get_evaluator(task: str):
    if task == "deepsearchqa":
        return _DEEPSEARCHQA_EVAL
    if task == "healthbench":
        return _HEALTHBENCH_EVAL
    if task == "researchrubrics":
        return _RESEARCHRUBRICS_EVAL
    return _BROWSECOMP_EVAL


def get_task_instruction(task: str) -> str:
    if task in ("browsecomp", "browsecomp-plus", "hle"):
        return BROWSECOMP_INSTRUCTION
    if task == "deepsearchqa":
        return DEEPSEARCHQA_INSTRUCTION
    return ""


__all__ = ["get_evaluator", "get_task_instruction"]
