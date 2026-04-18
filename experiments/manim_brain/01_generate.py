"""
Phase 1: Generate Manim animation variations for the same concept.

Concept: Pythagorean theorem (a² + b² = c²)
Variations:
  - v1_slow: slow pacing, one element at a time, long pauses
  - v2_fast: fast pacing, everything appears quickly
  - v3_visual: emphasis on geometric visualization, minimal text
  - v4_text_heavy: emphasis on equations and text, minimal geometry
  - v5_narrative: includes narration-style text overlays with storytelling

Each renders to an MP4, then we run Tribe to predict brain response.

Usage:
    source tribev2/.venv/bin/activate
    manim -pql experiments/manim_brain/01_generate.py V1Slow
    manim -pql experiments/manim_brain/01_generate.py V2Fast
    ... etc
    # Or render all:
    python experiments/manim_brain/01_generate.py
"""

from manim import *
import subprocess
import sys
from pathlib import Path

OUTPUT_DIR = Path("cache/manim_experiment")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class V1Slow(Scene):
    """Slow pacing, one element at a time, long pauses."""

    def construct(self):
        title = Text("The Pythagorean Theorem", font_size=48)
        self.play(Write(title), run_time=2)
        self.wait(2)
        self.play(FadeOut(title))
        self.wait(1)

        # Draw triangle slowly
        triangle = Polygon(ORIGIN, RIGHT * 3, RIGHT * 3 + UP * 4, color=BLUE)
        self.play(Create(triangle), run_time=3)
        self.wait(2)

        # Label sides one at a time
        a_label = MathTex("a = 3", font_size=36).next_to(triangle, DOWN)
        self.play(Write(a_label), run_time=1.5)
        self.wait(1.5)

        b_label = MathTex("b = 4", font_size=36).next_to(triangle, RIGHT)
        self.play(Write(b_label), run_time=1.5)
        self.wait(1.5)

        c_label = MathTex("c = 5", font_size=36).move_to(
            triangle.get_center() + LEFT * 0.8 + UP * 0.3
        )
        self.play(Write(c_label), run_time=1.5)
        self.wait(2)

        # Show equation
        eq = MathTex("a^2 + b^2 = c^2", font_size=48).to_edge(UP)
        self.play(Write(eq), run_time=2)
        self.wait(2)

        # Substitute
        eq2 = MathTex("3^2 + 4^2 = 5^2", font_size=48).next_to(eq, DOWN)
        self.play(Write(eq2), run_time=2)
        self.wait(1.5)

        eq3 = MathTex("9 + 16 = 25", font_size=48).next_to(eq2, DOWN)
        self.play(Write(eq3), run_time=2)
        self.wait(3)


class V2Fast(Scene):
    """Fast pacing, everything appears quickly."""

    def construct(self):
        title = Text("Pythagorean Theorem", font_size=48)
        self.play(Write(title), run_time=0.5)
        self.wait(0.3)
        self.play(FadeOut(title), run_time=0.3)

        # Triangle + labels all at once
        triangle = Polygon(ORIGIN, RIGHT * 3, RIGHT * 3 + UP * 4, color=BLUE)
        a_label = MathTex("a=3", font_size=32).next_to(triangle, DOWN)
        b_label = MathTex("b=4", font_size=32).next_to(triangle, RIGHT)
        c_label = MathTex("c=5", font_size=32).move_to(
            triangle.get_center() + LEFT * 0.8 + UP * 0.3
        )

        self.play(
            Create(triangle), Write(a_label), Write(b_label), Write(c_label),
            run_time=1
        )
        self.wait(0.5)

        # Equation chain fast
        eq1 = MathTex("a^2 + b^2 = c^2", font_size=44).to_edge(UP)
        eq2 = MathTex("3^2 + 4^2 = 5^2", font_size=44).next_to(eq1, DOWN)
        eq3 = MathTex("9 + 16 = 25 \\checkmark", font_size=44).next_to(eq2, DOWN)

        self.play(Write(eq1), run_time=0.5)
        self.play(Write(eq2), run_time=0.5)
        self.play(Write(eq3), run_time=0.5)
        self.wait(1)


class V3Visual(Scene):
    """Heavy geometric visualization, minimal text."""

    def construct(self):
        # Draw right triangle
        triangle = Polygon(ORIGIN, RIGHT * 3, RIGHT * 3 + UP * 4, color=WHITE)
        self.play(Create(triangle), run_time=1.5)
        self.wait(0.5)

        # Draw squares on each side
        sq_a = Square(side_length=3, color=RED, fill_opacity=0.3).next_to(triangle, DOWN, buff=0)
        sq_b = Square(side_length=4, color=GREEN, fill_opacity=0.3).next_to(triangle, RIGHT, buff=0)

        # Hypotenuse square (rotated)
        sq_c = Square(side_length=5, color=BLUE, fill_opacity=0.3)
        sq_c.rotate(np.arctan(4 / 3))
        sq_c.move_to(triangle.get_center() + LEFT * 1.5 + UP * 1)

        self.play(Create(sq_a), run_time=1)
        self.wait(0.5)

        a_area = MathTex("9", font_size=36, color=RED).move_to(sq_a)
        self.play(Write(a_area), run_time=0.5)

        self.play(Create(sq_b), run_time=1)
        self.wait(0.5)

        b_area = MathTex("16", font_size=36, color=GREEN).move_to(sq_b)
        self.play(Write(b_area), run_time=0.5)

        self.play(Create(sq_c), run_time=1)
        self.wait(0.5)

        c_area = MathTex("25", font_size=36, color=BLUE).move_to(sq_c)
        self.play(Write(c_area), run_time=0.5)
        self.wait(1)

        # Highlight: 9 + 16 = 25
        eq = MathTex("9", "+", "16", "=", "25", font_size=48).to_edge(UP)
        eq[0].set_color(RED)
        eq[2].set_color(GREEN)
        eq[4].set_color(BLUE)
        self.play(Write(eq), run_time=1)
        self.wait(2)


class V4TextHeavy(Scene):
    """Emphasis on equations and text, minimal geometry."""

    def construct(self):
        title = Text("The Pythagorean Theorem", font_size=42)
        subtitle = Text("For any right triangle:", font_size=28, color=GRAY).next_to(title, DOWN)
        self.play(Write(title), Write(subtitle), run_time=1.5)
        self.wait(1)
        self.play(FadeOut(title), FadeOut(subtitle))

        # Big equation
        eq = MathTex("a^2 + b^2 = c^2", font_size=72).to_edge(UP, buff=1)
        self.play(Write(eq), run_time=1.5)
        self.wait(1)

        # Text explanation
        lines = VGroup(
            Text("where:", font_size=28),
            MathTex("a", "\\text{ and }", "b", "\\text{ are the two shorter sides (legs)}", font_size=28),
            MathTex("c", "\\text{ is the longest side (hypotenuse)}", font_size=28),
        ).arrange(DOWN, aligned_edge=LEFT).next_to(eq, DOWN, buff=0.8)

        for line in lines:
            self.play(Write(line), run_time=1)
            self.wait(0.5)

        self.wait(1)

        # Example
        example_title = Text("Example:", font_size=32, color=YELLOW).to_edge(LEFT).shift(DOWN)
        self.play(Write(example_title), run_time=0.5)

        steps = VGroup(
            MathTex("a = 3, \\quad b = 4, \\quad c = ?", font_size=32),
            MathTex("3^2 + 4^2 = c^2", font_size=32),
            MathTex("9 + 16 = c^2", font_size=32),
            MathTex("25 = c^2", font_size=32),
            MathTex("c = 5", font_size=32, color=GREEN),
        ).arrange(DOWN, aligned_edge=LEFT).next_to(example_title, DOWN, aligned_edge=LEFT)

        for step in steps:
            self.play(Write(step), run_time=0.8)
            self.wait(0.5)

        self.wait(2)


class V5Narrative(Scene):
    """Storytelling approach with narration-style text."""

    def construct(self):
        # Hook
        hook = Text("2,500 years ago, a Greek mathematician\nmade a discovery...", font_size=32)
        self.play(Write(hook), run_time=2)
        self.wait(2)
        self.play(FadeOut(hook))

        name = Text("Pythagoras", font_size=56, color=YELLOW)
        self.play(Write(name), run_time=1)
        self.wait(1)
        self.play(FadeOut(name))

        # Discovery
        q = Text("He noticed something about right triangles...", font_size=28)
        self.play(Write(q), run_time=1.5)
        self.wait(1)
        self.play(FadeOut(q))

        # Show triangle
        triangle = Polygon(ORIGIN, RIGHT * 3, RIGHT * 3 + UP * 4, color=BLUE)
        self.play(Create(triangle), run_time=1.5)

        # Labels
        a_label = MathTex("3", font_size=36).next_to(triangle, DOWN)
        b_label = MathTex("4", font_size=36).next_to(triangle, RIGHT)
        c_label = MathTex("5", font_size=36).move_to(
            triangle.get_center() + LEFT * 0.8 + UP * 0.3
        )
        self.play(Write(a_label), Write(b_label), Write(c_label), run_time=1)
        self.wait(1)

        # Reveal
        reveal = Text("If you square the two short sides\nand add them together...", font_size=24).to_edge(UP)
        self.play(Write(reveal), run_time=1.5)
        self.wait(1)

        eq1 = MathTex("3^2 + 4^2 = 9 + 16 = 25", font_size=36).next_to(reveal, DOWN)
        self.play(Write(eq1), run_time=1)
        self.wait(1)

        reveal2 = Text("...you always get the square of the long side!", font_size=24, color=GREEN).next_to(eq1, DOWN)
        self.play(Write(reveal2), run_time=1.5)

        eq2 = MathTex("5^2 = 25 \\checkmark", font_size=36, color=GREEN).next_to(reveal2, DOWN)
        self.play(Write(eq2), run_time=1)
        self.wait(2)

        # General form
        self.play(*[FadeOut(mob) for mob in self.mobjects])
        final = MathTex("a^2 + b^2 = c^2", font_size=72, color=YELLOW)
        self.play(Write(final), run_time=1.5)
        self.wait(2)


def render_all():
    """Render all scenes to MP4."""
    scenes = ["V1Slow", "V2Fast", "V3Visual", "V4TextHeavy", "V5Narrative"]
    for scene in scenes:
        out = OUTPUT_DIR / f"{scene}.mp4"
        if out.exists():
            print(f"Cached: {scene}")
            continue
        print(f"Rendering: {scene}...")
        subprocess.run([
            sys.executable, "-m", "manim", "render",
            "-ql",  # low quality for speed
            "--media_dir", str(OUTPUT_DIR / "media"),
            __file__,
            scene,
        ], check=True)
        # Move rendered file
        rendered = list((OUTPUT_DIR / "media" / "videos" / "01_generate" / "480p15").glob(f"{scene}.mp4"))
        if rendered:
            rendered[0].rename(out)
            print(f"  -> {out}")


if __name__ == "__main__":
    render_all()
