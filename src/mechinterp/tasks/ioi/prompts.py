"""Prompt templates for the IOI task."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PromptTemplate:
    """Prompt template definition."""

    template_id: str
    template: str
    object_phrase: str

    def render(self, subject: str, indirect_object: str, giver: str) -> str:
        """Render the prompt."""
        return self.template.format(
            subject=subject,
            indirect_object=indirect_object,
            giver=giver,
            object_phrase=self.object_phrase,
        )


PROMPT_TEMPLATES: dict[str, PromptTemplate] = {
    "gave_object": PromptTemplate(
        template_id="gave_object",
        template=(
            "When {subject} and {indirect_object} went to the store, "
            "{giver} gave the {object_phrase} to"
        ),
        object_phrase="bag",
    ),
    "handed_object": PromptTemplate(
        template_id="handed_object",
        template=(
            "After {subject} and {indirect_object} met at the park, "
            "{giver} handed the {object_phrase} to"
        ),
        object_phrase="book",
    ),
    "brought_object": PromptTemplate(
        template_id="brought_object",
        template=(
            "When {subject} and {indirect_object} stayed after class, "
            "{giver} brought the {object_phrase} to"
        ),
        object_phrase="note",
    ),
    "passed_object": PromptTemplate(
        template_id="passed_object",
        template=(
            "After {subject} and {indirect_object} sat by the river, "
            "{giver} passed the {object_phrase} to"
        ),
        object_phrase="drink",
    ),
    "showed_object": PromptTemplate(
        template_id="showed_object",
        template=(
            "When {subject} and {indirect_object} waited by the station, "
            "{giver} showed the {object_phrase} to"
        ),
        object_phrase="map",
    ),
    "mailed_object": PromptTemplate(
        template_id="mailed_object",
        template=(
            "After {subject} and {indirect_object} finished at work, "
            "{giver} mailed the {object_phrase} to"
        ),
        object_phrase="letter",
    ),
    "read_object": PromptTemplate(
        template_id="read_object",
        template=(
            "When {subject} and {indirect_object} rested in the library, "
            "{giver} read the {object_phrase} to"
        ),
        object_phrase="story",
    ),
    "sent_object": PromptTemplate(
        template_id="sent_object",
        template=(
            "After {subject} and {indirect_object} talked after dinner, "
            "{giver} sent the {object_phrase} to"
        ),
        object_phrase="photo",
    ),
}


def get_template(template_id: str) -> PromptTemplate:
    """Resolve a template id to its prompt template."""

    try:
        return PROMPT_TEMPLATES[template_id]
    except KeyError as exc:
        raise ValueError(f"Unknown IOI template id '{template_id}'") from exc
