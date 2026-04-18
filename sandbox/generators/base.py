"""Generator protocol: any content source that produces a video file."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Generator(Protocol):
    """Interface for content generators.

    Any class that produces a video file from parameters can plug into
    the sandbox. The only contract: given params, return a video path.

    Implementations: ManimGenerator, TTS+image, diffusion models, etc.
    """

    name: str

    def param_space(self) -> dict[str, dict[str, Any]]:
        """Describe the parameter space.

        Returns dict mapping param names to specs::

            {
                "pacing": {"type": "float", "low": 0.5, "high": 3.0},
                "style":  {"type": "categorical", "choices": ["minimal", "rich"]},
            }
        """
        ...

    def generate(self, params: dict[str, Any], output_path: Path) -> Path:
        """Generate a video file from parameters.

        Args:
            params: Values within param_space bounds
            output_path: Where to write the .mp4

        Returns:
            Path to the generated video (may differ from output_path)
        """
        ...
