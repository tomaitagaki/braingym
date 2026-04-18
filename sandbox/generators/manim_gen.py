"""
ManimGenerator: parameterized Manim video generator.

Generates Manim scenes with controllable presentation parameters:
pacing, visual density, text density, reveal style, color scheme.

The generator writes a Manim script dynamically and renders it to MP4.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Any


class ManimGenerator:
    """Generate Manim educational animations with tunable presentation knobs.

    Args:
        concept: What to explain (used in title and content)
        quality: Manim quality flag — "l" (480p15), "m" (720p30), "h" (1080p60)
    """

    name = "manim"

    def __init__(self, concept: str = "Pythagorean Theorem", quality: str = "l"):
        self.concept = concept
        self.quality = quality

    def param_space(self) -> dict[str, dict[str, Any]]:
        return {
            "pacing": {
                "type": "float",
                "low": 0.3,
                "high": 3.0,
                "desc": "Animation speed multiplier (higher = slower, more deliberate)",
            },
            "pause_duration": {
                "type": "float",
                "low": 0.1,
                "high": 2.5,
                "desc": "Pause between elements in seconds",
            },
            "visual_density": {
                "type": "float",
                "low": 0.0,
                "high": 1.0,
                "desc": "Amount of geometric/visual content (0=minimal, 1=rich)",
            },
            "text_density": {
                "type": "float",
                "low": 0.0,
                "high": 1.0,
                "desc": "Amount of text/equation content (0=minimal, 1=heavy)",
            },
            "reveal_style": {
                "type": "categorical",
                "choices": ["sequential", "simultaneous", "progressive"],
                "desc": "How elements appear: one-by-one, all-at-once, or building up",
            },
            "color_scheme": {
                "type": "categorical",
                "choices": ["mono", "dual", "vibrant"],
                "desc": "Color variety: single color, two-tone, or full palette",
            },
            "narrative_hook": {
                "type": "categorical",
                "choices": ["none", "question", "story", "surprise"],
                "desc": "Opening style: jump straight in, pose a question, tell a story, or start with a surprising fact",
            },
        }

    def generate(self, params: dict[str, Any], output_path: Path) -> Path:
        """Generate a Manim video from parameters."""
        output_path = Path(output_path)
        if output_path.exists():
            return output_path

        script = self._build_script(params)

        with tempfile.TemporaryDirectory() as tmpdir:
            script_path = Path(tmpdir) / "scene.py"
            script_path.write_text(script)
            media_dir = Path(tmpdir) / "media"

            result = subprocess.run(
                [
                    sys.executable, "-m", "manim", "render",
                    f"-q{self.quality}",
                    "--media_dir", str(media_dir),
                    str(script_path),
                    "Generated",
                ],
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                raise RuntimeError(
                    f"Manim render failed:\n{result.stderr[-500:]}"
                )

            # Find the rendered file
            rendered = list(media_dir.rglob("Generated.mp4"))
            if not rendered:
                raise FileNotFoundError(
                    f"No Generated.mp4 found in {media_dir}"
                )

            output_path.parent.mkdir(parents=True, exist_ok=True)
            rendered[0].rename(output_path)

        return output_path

    def _build_script(self, params: dict[str, Any]) -> str:
        """Build a Manim Python script from parameters."""
        p = params
        pace = p.get("pacing", 1.0)
        pause = p.get("pause_duration", 1.0)
        vis = p.get("visual_density", 0.5)
        text = p.get("text_density", 0.5)
        reveal = p.get("reveal_style", "sequential")
        colors = p.get("color_scheme", "dual")
        hook = p.get("narrative_hook", "none")

        # Map color scheme to actual colors
        color_map = {
            "mono": ("BLUE", "BLUE", "BLUE"),
            "dual": ("BLUE", "RED", "GREEN"),
            "vibrant": ("YELLOW", "RED", "TEAL"),
        }
        c1, c2, c3 = color_map.get(colors, ("BLUE", "RED", "GREEN"))

        # Build the hook section
        hook_code = self._build_hook(hook, pace, pause)

        # Build content based on density balance
        visual_code = self._build_visuals(vis, pace, pause, reveal, c1, c2, c3)
        text_code = self._build_text(text, pace, pause, reveal)
        finale_code = self._build_finale(pace, pause, c1)

        return textwrap.dedent(f"""\
            from manim import *
            import numpy as np

            class Generated(Scene):
                def construct(self):
{textwrap.indent(hook_code, "                    ")}
{textwrap.indent(visual_code, "                    ")}
{textwrap.indent(text_code, "                    ")}
{textwrap.indent(finale_code, "                    ")}
        """)

    def _build_hook(self, hook: str, pace: float, pause: float) -> str:
        concept = self.concept
        if hook == "none":
            return f'title = Text("{concept}", font_size=48)\nself.play(Write(title), run_time={pace})\nself.wait({pause})\nself.play(FadeOut(title))\n'
        elif hook == "question":
            return (
                f'q = Text("What if there was a pattern\\nhidden in every right triangle?", font_size=32)\n'
                f'self.play(Write(q), run_time={pace * 1.5})\n'
                f'self.wait({pause * 1.5})\n'
                f'self.play(FadeOut(q))\n'
            )
        elif hook == "story":
            return (
                f'hook = Text("2,500 years ago, a mathematician\\nmade a discovery that changed geometry...", font_size=30)\n'
                f'self.play(Write(hook), run_time={pace * 2})\n'
                f'self.wait({pause * 1.5})\n'
                f'self.play(FadeOut(hook))\n'
            )
        elif hook == "surprise":
            return (
                f'hook = Text("Every right triangle hides\\na perfect equation.", font_size=36, color=YELLOW)\n'
                f'self.play(Write(hook), run_time={pace})\n'
                f'self.wait({pause})\n'
                f'self.play(FadeOut(hook))\n'
            )
        return ""

    def _build_visuals(
        self, density: float, pace: float, pause: float,
        reveal: str, c1: str, c2: str, c3: str,
    ) -> str:
        if density < 0.2:
            return "# minimal visuals\n"

        lines = [
            f'triangle = Polygon(ORIGIN, RIGHT * 3, RIGHT * 3 + UP * 4, color={c1})',
        ]

        if reveal == "simultaneous":
            lines += [
                f'a_label = MathTex("a=3", font_size=32).next_to(triangle, DOWN)',
                f'b_label = MathTex("b=4", font_size=32).next_to(triangle, RIGHT)',
                f'c_label = MathTex("c=5", font_size=32).move_to(triangle.get_center() + LEFT * 0.8 + UP * 0.3)',
                f'self.play(Create(triangle), Write(a_label), Write(b_label), Write(c_label), run_time={pace})',
                f'self.wait({pause})',
            ]
        elif reveal == "progressive":
            lines += [
                f'self.play(Create(triangle), run_time={pace})',
                f'a_label = MathTex("a=3", font_size=32).next_to(triangle, DOWN)',
                f'b_label = MathTex("b=4", font_size=32).next_to(triangle, RIGHT)',
                f'c_label = MathTex("c=5", font_size=32).move_to(triangle.get_center() + LEFT * 0.8 + UP * 0.3)',
                f'self.play(Write(a_label), run_time={pace * 0.5})',
                f'self.play(Write(b_label), run_time={pace * 0.5})',
                f'self.play(Write(c_label), run_time={pace * 0.5})',
                f'self.wait({pause})',
            ]
        else:  # sequential
            lines += [
                f'self.play(Create(triangle), run_time={pace * 1.5})',
                f'self.wait({pause})',
                f'a_label = MathTex("a=3", font_size=32).next_to(triangle, DOWN)',
                f'self.play(Write(a_label), run_time={pace})',
                f'self.wait({pause * 0.5})',
                f'b_label = MathTex("b=4", font_size=32).next_to(triangle, RIGHT)',
                f'self.play(Write(b_label), run_time={pace})',
                f'self.wait({pause * 0.5})',
                f'c_label = MathTex("c=5", font_size=32).move_to(triangle.get_center() + LEFT * 0.8 + UP * 0.3)',
                f'self.play(Write(c_label), run_time={pace})',
                f'self.wait({pause})',
            ]

        # High visual density: add squares on sides
        if density > 0.6:
            lines += [
                f'sq_a = Square(side_length=3, color={c2}, fill_opacity=0.3).next_to(triangle, DOWN, buff=0)',
                f'sq_b = Square(side_length=4, color={c3}, fill_opacity=0.3).next_to(triangle, RIGHT, buff=0)',
                f'self.play(Create(sq_a), run_time={pace})',
                f'a_area = MathTex("9", font_size=36, color={c2}).move_to(sq_a)',
                f'self.play(Write(a_area), run_time={pace * 0.5})',
                f'self.play(Create(sq_b), run_time={pace})',
                f'b_area = MathTex("16", font_size=36, color={c3}).move_to(sq_b)',
                f'self.play(Write(b_area), run_time={pace * 0.5})',
                f'self.wait({pause})',
            ]

        return "\n".join(lines) + "\n"

    def _build_text(
        self, density: float, pace: float, pause: float, reveal: str,
    ) -> str:
        if density < 0.2:
            return "# minimal text\n"

        lines = [
            f'eq = MathTex("a^2 + b^2 = c^2", font_size=48).to_edge(UP)',
            f'self.play(Write(eq), run_time={pace})',
            f'self.wait({pause})',
        ]

        if density > 0.4:
            lines += [
                f'eq2 = MathTex("3^2 + 4^2 = 5^2", font_size=44).next_to(eq, DOWN)',
                f'self.play(Write(eq2), run_time={pace})',
                f'self.wait({pause * 0.5})',
            ]

        if density > 0.6:
            lines += [
                f'eq3 = MathTex("9 + 16 = 25", font_size=44).next_to(eq2, DOWN)',
                f'self.play(Write(eq3), run_time={pace})',
                f'self.wait({pause})',
            ]

        if density > 0.8:
            lines += [
                f'note = Text("This works for ANY right triangle!", font_size=24, color=GRAY).to_edge(DOWN)',
                f'self.play(Write(note), run_time={pace})',
                f'self.wait({pause})',
            ]

        return "\n".join(lines) + "\n"

    def _build_finale(self, pace: float, pause: float, c1: str) -> str:
        return (
            f'self.play(*[FadeOut(mob) for mob in self.mobjects], run_time={pace * 0.5})\n'
            f'final = MathTex("a^2 + b^2 = c^2", font_size=72, color={c1})\n'
            f'self.play(Write(final), run_time={pace})\n'
            f'self.wait({pause * 1.5})\n'
        )
