"""Autokaggler package init."""

from .agent import main, build_success_result, AgentResult, TaskInput
from .data_manager import DataManager, DataMeta
from .pipeline import TitanicPipeline, TitanicPipelineResult

__all__ = [
    "main",
    "build_success_result",
    "AgentResult",
    "TaskInput",
    "DataManager",
    "DataMeta",
    "TitanicPipeline",
    "TitanicPipelineResult",
]
