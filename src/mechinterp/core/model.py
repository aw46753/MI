"""Shared TransformerBridge wrapper."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

from mechinterp.core.config import ExperimentConfig

if TYPE_CHECKING:
    import torch


def _import_transformer_bridge() -> Any:
    try:
        from transformer_lens.model_bridge import TransformerBridge
    except ImportError as exc:
        raise RuntimeError(
            "TransformerLens is not installed. Install dependencies with "
            "`pip install -r requirements.txt` or `pip install -e '.[dev]'`."
        ) from exc
    return TransformerBridge


def resolve_device(requested_device: str) -> str:
    """Validate a requested runtime device and return the normalized value."""

    device = requested_device.strip()
    if not device:
        raise ValueError("Device must be a non-empty string.")

    if device == "cpu":
        return device

    if device == "cuda" or device.startswith("cuda:"):
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("PyTorch is required to validate CUDA device availability.") from exc

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA device requested but CUDA is not available. "
                "Check that your runtime exposes an NVIDIA GPU, the NVIDIA driver is installed, "
                "and your PyTorch CUDA build matches the system CUDA stack."
            )
        return device

    raise ValueError(
        f"Unsupported device {device!r}. Expected 'cpu', 'cuda', or a device like 'cuda:0'."
    )


@dataclass
class ModelWrapper:
    """Thin wrapper around TransformerBridge used by tasks and experiments."""

    config: ExperimentConfig
    _model: Any | None = field(default=None, init=False, repr=False)

    def load(self) -> Any:
        """Boot the model lazily and return the underlying bridge."""

        if self._model is not None:
            return self._model

        TransformerBridge = _import_transformer_bridge()
        model = TransformerBridge.boot_transformers(self.config.model_name)
        resolved_device = resolve_device(self.config.device)

        try:
            model.enable_compatibility_mode(**self.config.compatibility_mode.__dict__)
        except AttributeError:
            pass

        try:
            model.to(resolved_device, print_details=False)
        except TypeError:
            model.to(resolved_device)
        except AttributeError:
            pass

        try:
            model.eval()
        except AttributeError:
            pass

        self._model = model
        return model

    @property
    def model(self) -> Any:
        """Return the booted model."""
        return self.load()

    def to_tokens(self, text: str | list[str], prepend_bos: bool = True) -> Any:
        """Convert text to tokens via the underlying model."""
        return self.model.to_tokens(text, prepend_bos=prepend_bos)

    def to_str_tokens(self, text: str | Any, prepend_bos: bool = True) -> list[str]:
        """Convert text or tokens to string tokens."""
        return list(self.model.to_str_tokens(text, prepend_bos=prepend_bos))

    def to_single_token(self, text: str) -> int:
        """Convert a single-token string to its token id."""
        return int(self.model.to_single_token(text))

    def forward_logits(self, inputs: Any, prepend_bos: bool = True) -> Any:
        """Run a forward pass and return logits."""
        tokens = inputs
        if isinstance(inputs, (str, list)):
            tokens = self.to_tokens(inputs, prepend_bos=prepend_bos)
        return self.model(tokens, return_type="logits")

    def run_with_cache(
        self,
        inputs: Any,
        *,
        names_filter: Any | None = None,
        stop_at_layer: int | None = None,
        return_type: str | None = "logits",
        prepend_bos: bool = True,
        **kwargs: Any,
    ) -> tuple[Any, Any]:
        """Run the model while caching activations."""

        tokens = inputs
        if isinstance(inputs, (str, list)):
            tokens = self.to_tokens(inputs, prepend_bos=prepend_bos)
        return self.model.run_with_cache(
            tokens,
            names_filter=names_filter,
            stop_at_layer=stop_at_layer,
            return_type=return_type,
            **kwargs,
        )

    def run_with_hooks(
        self,
        inputs: Any,
        *,
        fwd_hooks: list[tuple[Any, Any]] | None = None,
        return_type: str | None = "logits",
        prepend_bos: bool = True,
        **kwargs: Any,
    ) -> Any:
        """Run the model with temporary forward hooks."""

        tokens = inputs
        if isinstance(inputs, (str, list)):
            tokens = self.to_tokens(inputs, prepend_bos=prepend_bos)
        return self.model.run_with_hooks(
            tokens,
            fwd_hooks=fwd_hooks or [],
            return_type=return_type,
            **kwargs,
        )
