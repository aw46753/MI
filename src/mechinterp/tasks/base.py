"""Base task abstraction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Task(ABC):
    """Minimal task contract for reusable experiments."""

    name: str

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
    def default_hook_names(self, config: Any) -> list[str]:
        """Return default hook targets for the task."""
