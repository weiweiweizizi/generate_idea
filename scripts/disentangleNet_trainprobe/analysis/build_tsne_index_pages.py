#!/usr/bin/env python
"""
为患者级 t-SNE 输出构建可复用的索引页面。

此脚本功能：
- 读取已生成的 2D t-SNE combined-view PNG。
- 将其组合成 3x3 概览页面。
- 默认布局：
  - 行：All Patients、IMR Only、TT Only
  - 列：All Basis、No Side、Side Only
- 导出三张概览 PNG：
  - `combined_2d_index.png`
  - `usage_2d_index.png`
  - `activation_2d_index.png`

默认源文件夹：
- `tsne/all/all_basis`
- `tsne/all/no_side`
- `tsne/all/side_only`
- `tsne/imr/all_basis`
- `tsne/imr/no_side`
- `tsne/imr/side_only`
- `tsne/tt/all_basis`
- `tsne/tt/no_side`
- `tsne/tt/side_only`

用法：
`python scripts/disentangleNet_trainprobe/analysis/build_tsne_index_pages.py build`

结果输出到：
- `outputs/disentangleNet_trainprobe/v32_tri_region_masked_win20_e50/window_basis_activations_all/`
  `patient_pattern_analysis/tsne/index_pages/`
"""

from __future__ import annotations

import json
from pathlib import Path

import fire
from PIL import Image, ImageDraw, ImageFont


DEFAULT_ANALYSIS_ROOT = Path(
    "outputs/disentangleNet_trainprobe/v32_tri_region_masked_win20_e50/window_basis_activations_all/"
    "patient_pattern_analysis/tsne"
)


def load_font(size: int) -> ImageFont.ImageFont:
    """尝试加载 DejaVu / Liberation 字体，回退到默认字体"""
    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def fit_image(image: Image.Image, *, tile_width: int, tile_height: int) -> Image.Image:
    """
    将图像按比例缩放后居中放入 tile 区域。
    白色背景，保持宽高比，不变形。
    """
    image = image.convert("RGB")
    ratio = min(tile_width / image.width, tile_height / image.height)
    new_size = (max(1, int(image.width * ratio)), max(1, int(image.height * ratio)))
    resized = image.resize(new_size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (tile_width, tile_height), "white")
    offset = ((tile_width - resized.width) // 2, (tile_height - resized.height) // 2)
    canvas.paste(resized, offset)
    return canvas


def tile_path(root_dir: Path, feature_family: str) -> Path:
    """t-SNE combined PNG 的标准路径"""
    return root_dir / feature_family / "tsne_2d" / f"{feature_family}_tsne_2d_combined.png"


def build_page(
    *,
    feature_family: str,
    row_specs: list[tuple[str, list[Path]]],
    column_labels: list[str],
    output_path: Path,
    page_title: str,
    tile_width: int = 720,
    tile_height: int = 600,
) -> None:
    """
    构建一页 3x3 网格索引图。

    布局：左侧标签列（All Patients / IMR Only / TT Only）
          顶部标题 + 列标签
          主体为 tile 图片（带源目录标签）
    """
    title_font = load_font(34)
    label_font = load_font(24)
    small_font = load_font(18)

    rows = len(row_specs)
    cols = len(column_labels)
    left_label_width = 180
    top_title_height = 72
    col_label_height = 46
    outer_pad = 28
    gutter_x = 18
    gutter_y = 18

    width = outer_pad * 2 + left_label_width + cols * tile_width + (cols - 1) * gutter_x
    height = (
        outer_pad * 2
        + top_title_height
        + col_label_height
        + rows * tile_height
        + (rows - 1) * gutter_y
    )
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    # 页面标题
    draw.text((outer_pad, outer_pad), page_title, fill="black", font=title_font)

    grid_x0 = outer_pad + left_label_width
    grid_y0 = outer_pad + top_title_height + col_label_height

    # 列标签
    for col_idx, label in enumerate(column_labels):
        x = grid_x0 + col_idx * (tile_width + gutter_x) + tile_width / 2
        bbox = draw.textbbox((0, 0), label, font=label_font)
        text_w = bbox[2] - bbox[0]
        draw.text((x - text_w / 2, outer_pad + top_title_height), label, fill="black", font=label_font)

    # 行 + tiles
    for row_idx, (row_label, roots) in enumerate(row_specs):
        y = grid_y0 + row_idx * (tile_height + gutter_y)
        bbox = draw.textbbox((0, 0), row_label, font=label_font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        draw.text(
            (outer_pad + (left_label_width - text_w) / 2, y + tile_height / 2 - text_h / 2),
            row_label,
            fill="black",
            font=label_font,
        )

        for col_idx, root in enumerate(roots):
            x = grid_x0 + col_idx * (tile_width + gutter_x)
            source = tile_path(root, feature_family)
            tile = fit_image(Image.open(source), tile_width=tile_width, tile_height=tile_height)
            canvas.paste(tile, (x, y))
            draw.rectangle([x, y, x + tile_width, y + tile_height], outline="#bdbdbd", width=2)

            # tile 右下角标注源目录名
            short_root = root.name
            bbox = draw.textbbox((0, 0), short_root, font=small_font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            tag_x = x + tile_width - text_w - 12
            tag_y = y + tile_height - text_h - 10
            draw.rounded_rectangle(
                [tag_x - 8, tag_y - 4, x + tile_width - 6, y + tile_height - 6],
                radius=8,
                fill=(255, 255, 255),
                outline="#d0d0d0",
            )
            draw.text((tag_x, tag_y), short_root, fill="#444444", font=small_font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def build(
    analysis_root: str = str(DEFAULT_ANALYSIS_ROOT),
    output_dir: str = str(DEFAULT_ANALYSIS_ROOT / "index_pages"),
) -> None:
    """
    主入口函数：构建所有三张（combined / usage / activation）索引页面。

    参数：
    - `analysis_root`：t-SNE 输出根目录
    - `output_dir`：索引页面输出目录
    """
    analysis_root_path = Path(analysis_root)
    output_dir_path = Path(output_dir)

    # 3 列配置
    col_labels = ["All Basis", "No Side", "Side Only"]
    row_specs = [
        (
            "All Patients",
            [
                analysis_root_path / "all" / "all_basis",
                analysis_root_path / "all" / "no_side",
                analysis_root_path / "all" / "side_only",
            ],
        ),
        (
            "IMR Only",
            [
                analysis_root_path / "imr" / "all_basis",
                analysis_root_path / "imr" / "no_side",
                analysis_root_path / "imr" / "side_only",
            ],
        ),
        (
            "TT Only",
            [
                analysis_root_path / "tt" / "all_basis",
                analysis_root_path / "tt" / "no_side",
                analysis_root_path / "tt" / "side_only",
            ],
        ),
    ]

    summaries = []
    for feature_family in ("combined", "usage", "activation"):
        output_path = output_dir_path / f"{feature_family}_2d_index.png"
        build_page(
            feature_family=feature_family,
            row_specs=row_specs,
            column_labels=col_labels,
            output_path=output_path,
            page_title=f"{feature_family} t-SNE 2D Combined View Index",
        )
        summaries.append(
            {
                "feature_family": feature_family,
                "output_png": str(output_path.resolve()),
            }
        )

    summary = {
        "analysis_root": str(analysis_root_path.resolve()),
        "output_dir": str(output_dir_path.resolve()),
        "rows": [label for label, _ in row_specs],
        "columns": col_labels,
        "families": summaries,
    }
    output_dir_path.mkdir(parents=True, exist_ok=True)
    (output_dir_path / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    report_lines = [
        "# t-SNE Index Pages",
        "",
        f"- analysis_root: `{summary['analysis_root']}`",
        f"- output_dir: `{summary['output_dir']}`",
        f"- rows: `{', '.join(summary['rows'])}`",
        f"- columns: `{', '.join(summary['columns'])}`",
        "",
        "## Output Pages",
        "",
    ]
    for family in summaries:
        report_lines.append(f"- `{family['feature_family']}`: `{family['output_png']}`")
    (output_dir_path / "report.md").write_text("\n".join(report_lines).strip() + "\n", encoding="utf-8")


def main():
    fire.Fire({"build": build})


if __name__ == "__main__":
    main()
