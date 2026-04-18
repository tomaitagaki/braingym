"""
NarrationGenerator: parameterized TTS narration for brain-reward optimization.

Produces audio files from content blocks about a given topic.
Parameters control WHAT is said (content selection, structure) and
HOW it's said (voice, speed, pauses). Since text content drives ~85-90%
of predicted brain response, varying content IS varying brain signal.

Uses macOS `say` (no pip install, no API key, instant).
Tribe accepts audio_path directly — no video wrapping needed.
"""

from __future__ import annotations

import subprocess
import hashlib
from pathlib import Path
from typing import Any


# ============================================================
# Content pools — swap these for different topics
# ============================================================

PYTHAGOREAN_BLOCKS = {
    # Hooks (opening)
    "hook_question": (
        "Have you ever wondered why right triangles are so special? "
        "There's a hidden pattern that connects all three sides."
    ),
    "hook_historical": (
        "Over four thousand years ago, ancient Babylonian scribes "
        "carved a remarkable discovery into clay tablets."
    ),
    "hook_direct": (
        "Let's talk about the most famous theorem in all of mathematics."
    ),
    # Core content
    "definition": (
        "The Pythagorean theorem states that in any right triangle, "
        "the square of the longest side, called the hypotenuse, "
        "equals the sum of the squares of the other two sides."
    ),
    "equation": (
        "Written as an equation: a squared plus b squared equals c squared."
    ),
    "example_numeric": (
        "Take a triangle with sides three, four, and five. "
        "Three squared is nine. Four squared is sixteen. "
        "Nine plus sixteen equals twenty five. "
        "And indeed, five squared is twenty five."
    ),
    "example_visual": (
        "Imagine drawing a square on each side of a right triangle. "
        "The two smaller squares together have exactly the same area "
        "as the large square on the longest side."
    ),
    # Context
    "history": (
        "This theorem is named after the Greek mathematician Pythagoras, "
        "but it was known to the Babylonians more than a thousand years before him."
    ),
    "application": (
        "Engineers and architects use this theorem every day. "
        "Your phone's GPS uses it to calculate distances. "
        "Computer graphics rely on it to render three-dimensional scenes."
    ),
    # Closings
    "close_wonder": (
        "It's remarkable that such a simple equation holds true "
        "for every right triangle in the universe."
    ),
    "close_practical": (
        "Understanding this theorem is the first step toward "
        "trigonometry, calculus, and much of modern science."
    ),
    "close_challenge": (
        "Here's a challenge. Can you find a right triangle "
        "where all three sides are whole numbers, "
        "and the hypotenuse is less than ten?"
    ),
}


# ============================================================
# Narration structures — how blocks are assembled
# ============================================================

STRUCTURES = {
    "direct": ["definition", "equation", "example_numeric"],
    "story": ["hook_historical", "definition", "example_numeric", "history"],
    "hook_first": ["hook_question", "definition", "example_visual", "close_wonder"],
    "applied": ["hook_direct", "definition", "example_numeric", "application", "close_practical"],
    "deep": ["hook_question", "definition", "equation", "example_numeric", "example_visual", "history"],
    "minimal": ["definition", "example_numeric"],
    "challenge": ["hook_direct", "definition", "example_numeric", "close_challenge"],
}


class NarrationGenerator:
    """Generate narrated audio from parameterized content blocks.

    Uses macOS `say` for TTS. Tribe processes audio directly via audio_path.

    Args:
        content_blocks: Topic-specific content pool (default: Pythagorean theorem)
        structures: Named block orderings (default: built-in set)
    """

    name = "narration"

    def __init__(
        self,
        content_blocks: dict[str, str] | None = None,
        structures: dict[str, list[str]] | None = None,
    ):
        self.blocks = content_blocks or PYTHAGOREAN_BLOCKS
        self.structures = structures or STRUCTURES

    def param_space(self) -> dict[str, dict[str, Any]]:
        return {
            "structure": {
                "type": "categorical",
                "choices": list(self.structures.keys()),
                "desc": "Narrative structure (which blocks in what order)",
            },
            "voice": {
                "type": "categorical",
                "choices": ["Samantha", "Daniel", "Fred", "Flo"],
                "desc": "macOS TTS voice",
            },
            "speaking_rate": {
                "type": "float",
                "low": 140,
                "high": 220,
                "desc": "Words per minute (default ~175)",
            },
            "pause_between": {
                "type": "float",
                "low": 0.3,
                "high": 2.0,
                "desc": "Silence between content blocks (seconds)",
            },
            "include_closing": {
                "type": "categorical",
                "choices": ["none", "wonder", "practical", "challenge"],
                "desc": "Which closing to append (if not already in structure)",
            },
        }

    def generate(self, params: dict[str, Any], output_path: Path) -> Path:
        """Generate a narrated audio file (.wav) from parameters."""
        output_path = Path(output_path).with_suffix(".wav")
        if output_path.exists():
            return output_path

        # Assemble text from structure + optional closing
        structure_name = params.get("structure", "direct")
        block_keys = list(self.structures.get(structure_name, ["definition"]))

        closing = params.get("include_closing", "none")
        if closing != "none":
            close_key = f"close_{closing}"
            if close_key in self.blocks and close_key not in block_keys:
                block_keys.append(close_key)

        text_segments = [self.blocks[k] for k in block_keys if k in self.blocks]

        # Generate audio
        voice = params.get("voice", "Samantha")
        rate = int(params.get("speaking_rate", 175))
        pause = float(params.get("pause_between", 0.8))

        output_path.parent.mkdir(parents=True, exist_ok=True)

        if len(text_segments) == 1 or pause < 0.05:
            # Single pass — no silence insertion needed
            full_text = " ".join(text_segments)
            self._say(full_text, voice, rate, output_path)
        else:
            # Render each segment, insert silence, concatenate
            self._render_with_pauses(text_segments, voice, rate, pause, output_path)

        return output_path

    def _say(self, text: str, voice: str, rate: int, output_path: Path) -> None:
        """Render text to audio via macOS say."""
        aiff = output_path.with_suffix(".aiff")
        subprocess.run(
            ["say", "-v", voice, "-r", str(rate), "-o", str(aiff), text],
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(aiff), "-ar", "16000", "-ac", "1", str(output_path)],
            check=True,
            capture_output=True,
        )
        aiff.unlink(missing_ok=True)

    def _render_with_pauses(
        self,
        segments: list[str],
        voice: str,
        rate: int,
        pause: float,
        output_path: Path,
    ) -> None:
        """Render segments with silence between them."""
        tmp_dir = output_path.parent / f"_tmp_{output_path.stem}"
        tmp_dir.mkdir(exist_ok=True)

        segment_files = []
        for i, text in enumerate(segments):
            seg_path = tmp_dir / f"seg_{i:02d}.wav"
            self._say(text, voice, rate, seg_path)
            segment_files.append(seg_path)

        # Build ffmpeg concat filter with silence
        # Create silence file
        silence_path = tmp_dir / "silence.wav"
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-f", "lavfi", "-i", f"anullsrc=r=16000:cl=mono",
                "-t", str(pause),
                str(silence_path),
            ],
            check=True,
            capture_output=True,
        )

        # Build concat list
        concat_list = tmp_dir / "concat.txt"
        lines = []
        for i, sf in enumerate(segment_files):
            lines.append(f"file '{sf.name}'")
            if i < len(segment_files) - 1:
                lines.append(f"file '{silence_path.name}'")
        concat_list.write_text("\n".join(lines))

        subprocess.run(
            [
                "ffmpeg", "-y",
                "-f", "concat", "-safe", "0",
                "-i", str(concat_list),
                "-c", "copy",
                str(output_path),
            ],
            check=True,
            capture_output=True,
        )

        # Cleanup
        for f in tmp_dir.iterdir():
            f.unlink()
        tmp_dir.rmdir()

    def describe_params(self, params: dict[str, Any]) -> str:
        structure = params.get("structure", "direct")
        blocks = self.structures.get(structure, [])
        closing = params.get("include_closing", "none")
        voice = params.get("voice", "Samantha")
        rate = params.get("speaking_rate", 175)
        return (
            f"{structure} structure ({len(blocks)} blocks), "
            f"closing={closing}, voice={voice}, rate={rate:.0f} wpm"
        )
