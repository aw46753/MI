"""Synthetic IOI dataset generation."""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from typing import Any

from mechinterp.core.config import ExperimentConfig
from mechinterp.tasks.ioi.prompts import get_template


@dataclass(frozen=True)
class IOIExample:
    """One synthetic IOI prompt."""

    prompt: str
    correct_token: str
    wrong_token: str
    subject: str
    indirect_object: str
    template_id: str
    split: str

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable representation."""
        return asdict(self)


def _split_pools(config: ExperimentConfig) -> dict[str, dict[str, list[str]]]:
    """Partition names and templates into standard and shifted pools."""

    rng = random.Random(config.seed)

    names = list(config.dataset.names)
    templates = list(config.dataset.templates)
    rng.shuffle(names)
    rng.shuffle(templates)

    shifted_names = names[: config.dataset.shifted_name_count]
    standard_names = names[config.dataset.shifted_name_count :]
    shifted_templates = templates[: config.dataset.shifted_template_count]
    standard_templates = templates[config.dataset.shifted_template_count :]

    return {
        "standard": {"names": standard_names, "templates": standard_templates},
        "shifted": {"names": shifted_names, "templates": shifted_templates},
    }


def render_ioi_prompt(template_id: str, subject: str, indirect_object: str, giver: str | None = None) -> str:
    """Render an IOI prompt for the given names and template."""

    template = get_template(template_id)
    return template.render(subject=subject, indirect_object=indirect_object, giver=giver or subject)


def build_ioi_dataset(split: str, config: ExperimentConfig) -> list[IOIExample]:
    """Build a deterministic IOI dataset for one split."""

    pools = _split_pools(config)
    if split not in pools:
        raise ValueError(f"Unknown split '{split}'")

    name_pool = pools[split]["names"]
    template_pool = pools[split]["templates"]
    size = config.dataset.dataset_sizes[split]

    if len(name_pool) < 2:
        raise ValueError(f"Split '{split}' must have at least two names")
    if len(template_pool) < 1:
        raise ValueError(f"Split '{split}' must have at least one template")

    candidates: list[tuple[str, str, str]] = []
    for template_id in template_pool:
        for subject in name_pool:
            for indirect_object in name_pool:
                if subject == indirect_object:
                    continue
                candidates.append((template_id, subject, indirect_object))

    if size > len(candidates):
        raise ValueError(
            f"Requested {size} examples for split '{split}', but only {len(candidates)} unique combinations are available"
        )

    rng = random.Random(f"{config.seed}:{split}")
    rng.shuffle(candidates)

    examples: list[IOIExample] = []
    for template_id, subject, indirect_object in candidates[:size]:
        prompt = render_ioi_prompt(template_id, subject=subject, indirect_object=indirect_object)
        examples.append(
            IOIExample(
                prompt=prompt,
                correct_token=f" {indirect_object}",
                wrong_token=f" {subject}",
                subject=subject,
                indirect_object=indirect_object,
                template_id=template_id,
                split=split,
            )
        )

    return examples
