from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, Sequence
from .models import PipelineState, RequestContext

class FastPathStage(Protocol):
    name: str
    def run(self, request: RequestContext, state: PipelineState) -> PipelineState: ...

class AsyncEventPublisher(Protocol):
    def publish(self, request: RequestContext, state: PipelineState) -> None: ...

@dataclass(frozen=True, slots=True)
class FastPathPipeline:
    stages: Sequence[FastPathStage]
    def __post_init__(self) -> None:
        names = [stage.name for stage in self.stages]
        if len(names) != len(set(names)): raise ValueError("fast-path stage names must be unique")
    def execute(self, request: RequestContext, initial: PipelineState | None = None) -> PipelineState:
        state = initial or PipelineState()
        for stage in self.stages: state = stage.run(request, state)
        return state