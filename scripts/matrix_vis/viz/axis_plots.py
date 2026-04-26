from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


PANEL_BG = (255, 255, 255)
GRID_COLOR = (220, 220, 220)
AXIS_COLOR = (80, 80, 80)
TEXT_COLOR = (30, 30, 30)
SOLVE_COLOR = (220, 70, 70)
GT_COLOR = (60, 110, 220)


def _to_canvas_points(
    x_values: np.ndarray,
    y_values: np.ndarray,
    *,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    left: int,
    top: int,
    width: int,
    height: int,
) -> list[tuple[int, int]]:
    x_span = max(x_max - x_min, 1e-6)
    y_span = max(y_max - y_min, 1e-6)
    points: list[tuple[int, int]] = []
    for x_val, y_val in zip(x_values.tolist(), y_values.tolist()):
        px = left + int(round((x_val - x_min) / x_span * width))
        py = top + height - int(round((y_val - y_min) / y_span * height))
        points.append((px, py))
    return points


def _draw_grid(
    draw: ImageDraw.ImageDraw,
    *,
    left: int,
    top: int,
    width: int,
    height: int,
    x_ticks: int = 4,
    y_ticks: int = 4,
) -> None:
    for idx in range(x_ticks + 1):
        x = left + int(round(width * idx / max(x_ticks, 1)))
        draw.line([(x, top), (x, top + height)], fill=GRID_COLOR, width=1)
    for idx in range(y_ticks + 1):
        y = top + int(round(height * idx / max(y_ticks, 1)))
        draw.line([(left, y), (left + width, y)], fill=GRID_COLOR, width=1)
    draw.rectangle([left, top, left + width, top + height], outline=AXIS_COLOR, width=1)


def save_axis_trajectory_plot(
    *,
    output_dir: Path,
    time_grid: np.ndarray,
    trajectory: np.ndarray,
    point_ids: np.ndarray,
    axis: str,
) -> str | None:
    width, height = 1200, 720
    margin_left, margin_top = 80, 60
    plot_width, plot_height = 1040, 580
    image = Image.new("RGB", (width, height), color=PANEL_BG)
    draw = ImageDraw.Draw(image)

    _draw_grid(
        draw,
        left=margin_left,
        top=margin_top,
        width=plot_width,
        height=plot_height,
    )
    draw.text((margin_left, 20), f"{axis.upper()}-axis trajectories", fill=TEXT_COLOR)

    y_min = float(np.min(trajectory))
    y_max = float(np.max(trajectory))
    if abs(y_max - y_min) < 1e-6:
        y_min -= 1.0
        y_max += 1.0

    for idx, point_id in enumerate(point_ids.tolist()):
        color = (
            int((37 * idx) % 180 + 50),
            int((71 * idx) % 180 + 50),
            int((103 * idx) % 180 + 50),
        )
        points = _to_canvas_points(
            time_grid,
            trajectory[idx],
            x_min=float(time_grid.min()),
            x_max=float(time_grid.max()),
            y_min=y_min,
            y_max=y_max,
            left=margin_left,
            top=margin_top,
            width=plot_width,
            height=plot_height,
        )
        draw.line(points, fill=color, width=2)
        draw.text((margin_left + 10, margin_top + 10 + idx * 14), f"p{point_id}", fill=color)

    image.save(output_dir / "axis_trajectory.png")
    return None


def save_axis_ground_truth_comparison_plot(
    *,
    output_dir: Path,
    time_grid: np.ndarray,
    reconstructed: np.ndarray,
    ground_truth: np.ndarray,
    point_ids: np.ndarray,
    axis: str,
) -> str | None:
    if reconstructed.shape != ground_truth.shape:
        return "ground truth shape does not match reconstructed trajectory; skipped comparison plot"

    num_points = reconstructed.shape[0]
    cols = 4
    rows = int(np.ceil(num_points / cols))
    panel_w, panel_h = 280, 180
    outer_pad = 24
    title_h = 30
    image = Image.new(
        "RGB",
        (
            outer_pad * 2 + cols * panel_w,
            outer_pad * 2 + title_h + rows * panel_h,
        ),
        color=PANEL_BG,
    )
    draw = ImageDraw.Draw(image)
    draw.text(
        (outer_pad, outer_pad),
        f"{axis.upper()}-axis reconstructed vs ground truth",
        fill=TEXT_COLOR,
    )

    for idx, point_id in enumerate(point_ids.tolist()):
        row = idx // cols
        col = idx % cols
        left = outer_pad + col * panel_w + 12
        top = outer_pad + title_h + row * panel_h + 18
        width = panel_w - 24
        height = panel_h - 42
        _draw_grid(draw, left=left, top=top, width=width, height=height, x_ticks=3, y_ticks=3)
        draw.text((left, top - 16), f"p{point_id}", fill=TEXT_COLOR)

        y_values = np.concatenate([reconstructed[idx], ground_truth[idx]], axis=0)
        y_min = float(y_values.min())
        y_max = float(y_values.max())
        if abs(y_max - y_min) < 1e-6:
            y_min -= 1.0
            y_max += 1.0

        gt_points = _to_canvas_points(
            time_grid,
            ground_truth[idx],
            x_min=float(time_grid.min()),
            x_max=float(time_grid.max()),
            y_min=y_min,
            y_max=y_max,
            left=left,
            top=top,
            width=width,
            height=height,
        )
        rec_points = _to_canvas_points(
            time_grid,
            reconstructed[idx],
            x_min=float(time_grid.min()),
            x_max=float(time_grid.max()),
            y_min=y_min,
            y_max=y_max,
            left=left,
            top=top,
            width=width,
            height=height,
        )
        draw.line(gt_points, fill=GT_COLOR, width=2)
        draw.line(rec_points, fill=SOLVE_COLOR, width=2)

    legend_y = image.height - 20
    draw.line([(outer_pad, legend_y), (outer_pad + 24, legend_y)], fill=GT_COLOR, width=2)
    draw.text((outer_pad + 30, legend_y - 8), "ground truth", fill=TEXT_COLOR)
    draw.line([(outer_pad + 150, legend_y), (outer_pad + 174, legend_y)], fill=SOLVE_COLOR, width=2)
    draw.text((outer_pad + 180, legend_y - 8), "reconstructed", fill=TEXT_COLOR)
    image.save(output_dir / "axis_ground_truth_comparison.png")
    return None
