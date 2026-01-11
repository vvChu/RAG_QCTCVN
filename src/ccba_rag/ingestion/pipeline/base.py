import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class PipelineContext:
    """Shared context for the pipeline execution."""
    file_path: str
    metadata: Dict[str, Any] = None
    stats: Dict[str, Any] = None
    errors: List[str] = None

    def __post_init__(self):
        if self.metadata is None: self.metadata = {}
        if self.stats is None: self.stats = {}
        if self.errors is None: self.errors = []

class PipelineStage(ABC):
    """Abstract base class for a pipeline stage."""

    @abstractmethod
    async def run(self, context: PipelineContext, payload: Any) -> Any:
        """
        Process the payload and return result for next stage.

        Args:
            context: Shared pipeline context (metadata, stats)
            payload: Input data from previous stage

        Returns:
            Output data for next stage
        """
        pass

class IngestionPipeline:
    """Orchestrator for the ingestion process."""

    def __init__(self, stages: List[PipelineStage]):
        self.stages = stages

    async def run(self, file_path: str) -> PipelineContext:
        """Run the full pipeline for a file."""
        context = PipelineContext(file_path=file_path)
        payload = file_path # Initial payload

        try:
            for stage in self.stages:
                # Update payload with result of stage
                payload = await stage.run(context, payload)

                # If stage returns None, stop pipeline (e.g., filtered out)
                if payload is None:
                    break

        except Exception as e:
            context.errors.append(str(e))
            # Log error here

        return context
