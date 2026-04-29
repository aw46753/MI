"""Synthetic subject-verb agreement dataset generation."""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from typing import Any

from mechinterp.core.config import ExperimentConfig


NOUNS = {
    "singular": [" dog", " cat", " child", " bird", " singer", " doctor", " farmer", " pilot", " teacher", " poet"],
    "plural": [" dogs", " cats", " children", " birds", " singers", " doctors", " farmers", " pilots", " teachers", " poets"],
}
ATTRACTORS = {
    "singular": [" teacher", " pilot", " baker", " artist", " guard", " painter", " driver", " clerk"],
    "plural": [" teachers", " pilots", " bakers", " artists", " guards", " painters", " drivers", " clerks"],
}
CONTEXTS = [
    " near the barn",
    " by the window",
    " after the storm",
    " in the garden",
    " near the station",
    " beyond the fence",
    " at the market",
    " before the concert",
    " under the bridge",
    " beside the cabin",
]
INTRODUCTIONS = ["Statement", "Report", "Observation", "Note"]
ATTRACTOR_LINKS = [" beside", " near", " behind", " across from"]


@dataclass(frozen=True)
class SVAExample:
    """One synthetic subject-verb agreement prompt."""

    prompt: str
    corrupted_prompt: str
    correct_token: str
    wrong_token: str
    subject: str
    subject_number: str
    attractor: str | None
    attractor_number: str | None
    verb_singular: str
    verb_plural: str
    context: str
    intro: str
    attractor_link: str | None
    split: str

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable representation."""
        return asdict(self)


def render_sva_prompt(
    subject: str,
    *,
    context: str,
    attractor: str | None = None,
    intro: str = "Statement",
    attractor_link: str = " beside",
) -> str:
    """Render an SVA prompt ending immediately before the verb."""

    if attractor is None:
        return f"{intro}: The{subject}{context}"
    return f"{intro}: The{subject}{attractor_link} the{attractor}{context}"


def _opposite_number(number: str) -> str:
    return "plural" if number == "singular" else "singular"


def build_sva_dataset(split: str, config: ExperimentConfig) -> list[SVAExample]:
    """Build a deterministic SVA dataset for one split."""

    if split not in {"standard", "shifted"}:
        raise ValueError(f"Unknown split '{split}'")

    size = config.dataset.dataset_sizes[split]
    rng = random.Random(f"{config.seed}:{split}:sva")
    candidates: list[tuple[str, str, str, str | None, str, str | None]] = []
    for subject_number in ("singular", "plural"):
        for subject in NOUNS[subject_number]:
            for context in CONTEXTS:
                if split == "standard":
                    for intro in INTRODUCTIONS:
                        candidates.append((subject_number, subject, context, None, intro, None))
                else:
                    attractor_number = _opposite_number(subject_number)
                    for attractor in ATTRACTORS[attractor_number]:
                        for intro in INTRODUCTIONS:
                            for attractor_link in ATTRACTOR_LINKS:
                                candidates.append((subject_number, subject, context, attractor, intro, attractor_link))

    if size > len(candidates):
        raise ValueError(
            f"Requested {size} examples for split '{split}', but only {len(candidates)} unique combinations are available"
        )

    rng.shuffle(candidates)
    examples: list[SVAExample] = []
    for subject_number, subject, context, attractor, intro, attractor_link in candidates[:size]:
        corrupted_subject_number = _opposite_number(subject_number)
        corrupted_subject = NOUNS[corrupted_subject_number][
            NOUNS[subject_number].index(subject) % len(NOUNS[corrupted_subject_number])
        ]
        attractor_number = None if attractor is None else corrupted_subject_number

        prompt = render_sva_prompt(
            subject,
            context=context,
            attractor=attractor,
            intro=intro,
            attractor_link=attractor_link or " beside",
        )
        corrupted_prompt = render_sva_prompt(
            subject=corrupted_subject,
            context=context,
            attractor=attractor,
            intro=intro,
            attractor_link=attractor_link or " beside",
        )
        examples.append(
            SVAExample(
                prompt=prompt,
                corrupted_prompt=corrupted_prompt,
                correct_token=" is" if subject_number == "singular" else " are",
                wrong_token=" are" if subject_number == "singular" else " is",
                subject=subject,
                subject_number=subject_number,
                attractor=attractor,
                attractor_number=attractor_number,
                verb_singular=" is",
                verb_plural=" are",
                context=context,
                intro=intro,
                attractor_link=attractor_link,
                split=split,
            )
        )

    return examples
