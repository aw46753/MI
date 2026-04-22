"""Base task abstraction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Task(ABC):
    """Minimal task contract for reusable experiments."""

    name: str

    @abstractmethod
    def split_names(self, config: Any) -> list[str]:
        """Return the dataset split names used by behavior experiments."""

    @abstractmethod
    def build_behavior_split(self, model: Any, split: str, config: Any) -> dict[str, Any]:
        """Build behavior rows for one split."""

    @abstractmethod
    def build_dataset(self, split: str, config: Any) -> list[Any]:
        """Build task examples for a split."""

    @abstractmethod
    def score_example(self, model: Any, example: Any) -> Any:
        """Score one example."""

    @abstractmethod
    def make_pairs(self, dataset: list[Any], scored_examples: list[Any], model: Any) -> list[Any]:
        """Build clean/corrupted pairs for patching."""

    @abstractmethod
    def build_error_pairs(
        self,
        rows: list[dict[str, Any]],
        source_error_type: str,
        target_error_type: str,
        model: Any,
    ) -> list[Any]:
        """Build matched error pairs for patching and analysis."""

    @abstractmethod
    def default_hook_names(self, config: Any) -> list[str]:
        """Return default hook targets for the task."""

    def supports_patching(self) -> bool:
        """Return whether the task supports activation patching."""
        return True
