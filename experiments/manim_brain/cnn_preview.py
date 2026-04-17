"""
Preview: CNN explanation in Manim. Uses Text() only (no LaTeX needed).
Render with: manim render -ql experiments/manim_brain/cnn_preview.py CNNExplain
"""

from manim import *
import numpy as np


class CNNExplain(Scene):
    """Explain how a CNN processes an image — mid-range params for preview."""

    def construct(self):
        # === HOOK ===
        title = Text("How does a CNN see an image?", font_size=36, color=YELLOW)
        self.play(Write(title), run_time=1.5)
        self.wait(1.0)
        self.play(FadeOut(title))

        # === INPUT IMAGE (5x5 pixel grid) ===
        input_label = Text("Input Image (5x5)", font_size=24).to_edge(UP)
        self.play(Write(input_label), run_time=0.8)

        pixel_values = [
            [0, 1, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 1, 1, 1, 0],
        ]

        cell_size = 0.5
        input_grid = VGroup()
        for r in range(5):
            for c in range(5):
                val = pixel_values[r][c]
                sq = Square(side_length=cell_size)
                sq.set_fill(WHITE if val == 1 else DARK_GRAY, opacity=0.8)
                sq.set_stroke(GRAY, width=1)
                sq.move_to(np.array([c * cell_size, -r * cell_size, 0]))
                input_grid.add(sq)

        input_grid.move_to(LEFT * 4)
        self.play(Create(input_grid), run_time=1.5)
        self.wait(0.8)

        # === CONVOLUTION FILTER (3x3) ===
        filter_label = Text("3x3 Filter", font_size=20, color=BLUE).next_to(input_grid, DOWN, buff=0.5)
        self.play(Write(filter_label), run_time=0.6)

        # Highlight 3x3 region sliding across input
        highlight = Square(side_length=cell_size * 3, color=BLUE, stroke_width=3)
        highlight.move_to(input_grid[0:3].get_center())  # top-left 3x3
        # Position at top-left of grid
        top_left = input_grid[0].get_center()
        highlight.move_to(top_left + np.array([cell_size, -cell_size, 0]))
        self.play(Create(highlight), run_time=0.5)

        # Slide filter across (3 positions horizontally)
        arrow_text = Text("Slide filter across image", font_size=18, color=GRAY).to_edge(DOWN)
        self.play(Write(arrow_text), run_time=0.5)

        for dx in range(3):
            for dy in range(3):
                target = top_left + np.array([
                    (dx + 1) * cell_size,
                    -(dy + 1) * cell_size,
                    0,
                ])
                self.play(highlight.animate.move_to(target), run_time=0.3)

        self.play(FadeOut(highlight), FadeOut(arrow_text), FadeOut(filter_label), run_time=0.5)
        self.wait(0.5)

        # === FEATURE MAP OUTPUT ===
        feature_label = Text("Feature Map (3x3)", font_size=20, color=GREEN).move_to(UP * 2.5 + RIGHT * 0.5)
        self.play(Write(feature_label), run_time=0.6)

        feat_values = [[2, 3, 1], [1, 3, 1], [1, 3, 2]]
        feature_grid = VGroup()
        for r in range(3):
            for c in range(3):
                val = feat_values[r][c]
                sq = Square(side_length=cell_size * 0.8)
                intensity = val / 3.0
                sq.set_fill(GREEN, opacity=intensity)
                sq.set_stroke(GREEN, width=1)
                num = Text(str(val), font_size=14).move_to(sq)
                cell = VGroup(sq, num)
                cell.move_to(np.array([c * cell_size, -r * cell_size, 0]))
                feature_grid.add(cell)

        feature_grid.move_to(RIGHT * 0 + DOWN * 0.3)

        # Arrow from input to feature map
        arrow1 = Arrow(input_grid.get_right(), feature_grid.get_left(), buff=0.3, color=BLUE)
        conv_label = Text("Conv", font_size=16, color=BLUE).next_to(arrow1, UP, buff=0.1)
        self.play(Create(arrow1), Write(conv_label), run_time=0.8)
        self.play(Create(feature_grid), run_time=1.0)
        self.wait(0.8)

        # === POOLING ===
        pool_label = Text("Max Pool (2x2)", font_size=20, color=RED).move_to(UP * 2.5 + RIGHT * 4)
        self.play(Write(pool_label), run_time=0.6)

        pool_grid = VGroup()
        pool_vals = [[3, 3], [3, 3]]  # max of each 2x2 region (simplified)
        for r in range(2):
            for c in range(2):
                val = pool_vals[r][c]
                sq = Square(side_length=cell_size * 0.8)
                sq.set_fill(RED, opacity=val / 3.0)
                sq.set_stroke(RED, width=1)
                num = Text(str(val), font_size=14).move_to(sq)
                cell = VGroup(sq, num)
                cell.move_to(np.array([c * cell_size, -r * cell_size, 0]))
                pool_grid.add(cell)

        pool_grid.move_to(RIGHT * 4 + DOWN * 0.3)

        arrow2 = Arrow(feature_grid.get_right(), pool_grid.get_left(), buff=0.3, color=RED)
        pool_op = Text("Pool", font_size=16, color=RED).next_to(arrow2, UP, buff=0.1)
        self.play(Create(arrow2), Write(pool_op), run_time=0.8)
        self.play(Create(pool_grid), run_time=1.0)
        self.wait(0.8)

        # === CLASSIFICATION OUTPUT ===
        self.play(
            *[FadeOut(m) for m in [input_label, feature_label, pool_label]],
            run_time=0.5,
        )

        output_label = Text("Output: Classification", font_size=24).to_edge(DOWN, buff=1.5)
        classes = VGroup(
            Text("cat: 12%", font_size=18, color=GRAY),
            Text("dog: 8%", font_size=18, color=GRAY),
            Text("digit '1': 80%", font_size=18, color=GREEN),
        ).arrange(DOWN, aligned_edge=LEFT).next_to(output_label, UP, buff=0.3)

        arrow3 = Arrow(pool_grid.get_bottom(), classes.get_top(), buff=0.2, color=WHITE)
        fc_label = Text("Fully Connected", font_size=14).next_to(arrow3, RIGHT, buff=0.1)
        self.play(Create(arrow3), Write(fc_label), run_time=0.8)
        self.play(Write(output_label), run_time=0.5)
        self.play(Write(classes), run_time=1.0)
        self.wait(1.0)

        # === FINALE ===
        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=0.8)
        summary = Text("Input → Conv → Pool → Classify", font_size=36, color=YELLOW)
        self.play(Write(summary), run_time=1.5)
        self.wait(1.5)
