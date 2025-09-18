"""AutoKaggler package init re-exporting key orchestrator objects."""

from .agent import AgentResult, TaskInput, build_success_result, main
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
