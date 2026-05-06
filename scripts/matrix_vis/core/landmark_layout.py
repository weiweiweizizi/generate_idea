from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import yaml

# 从配置文件中加载面部区域定义和对称点对，返回一个字典和一个列表
# 配置文件的内容示例：
# face_regions:
#   left_eye: 
#   right_eye: 
# symmetric_pairs:

def _parse_symmetric_pairs(raw: Any) -> list[tuple[int, int]]:
    if isinstance(raw, list) and raw and isinstance(raw[0], (list, tuple)):
        return [(int(left), int(right)) for left, right in raw]
    if isinstance(raw, list) and raw and isinstance(raw[0], str):
        joined = f"[{', '.join(str(item) for item in raw)}]"
        return [(int(left), int(right)) for left, right in ast.literal_eval(joined)]
    if isinstance(raw, str):
        return [(int(left), int(right)) for left, right in ast.literal_eval(raw)]
    return []

# 从yaml加载面部区域配置（face_regions）
def load_face_region_config(
    config_path: str | Path,
    *,
    extractor_name: str = "mediapipe",
) -> tuple[dict[str, list[int]], list[tuple[int, int]]]:
    path = Path(config_path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    if not isinstance(raw, dict):
        raise ValueError(f"Expected extractor config mapping at {path}")
    section = raw.get(extractor_name)
    if not isinstance(section, dict):
        raise ValueError(f"Missing extractor section {extractor_name!r} in {path}")

    face_regions_raw = section.get("face_regions")
    if not isinstance(face_regions_raw, dict):
        raise ValueError(f"Missing or invalid face_regions in {path}")
    face_regions = {
        str(region): [int(point_id) for point_id in point_ids]
        for region, point_ids in face_regions_raw.items()
    }
    symmetric_pairs = _parse_symmetric_pairs(section.get("symmetric_pairs", []))
    return face_regions, symmetric_pairs

# 将各区域按照配置中的顺序进行分组，并将对称区域相邻放，返回一个包含所有点ID的元组
def build_grouped_face_region_subset(
    face_regions: dict[str, list[int]],
    symmetric_pairs: list[tuple[int, int]],
) -> tuple[int, ...]:
    sym_map: dict[int, int] = {}
    for left, right in symmetric_pairs:
        sym_map[int(left)] = int(right)
        sym_map[int(right)] = int(left)

    ordered: list[int] = []
    seen: set[int] = set()
    for left_points in face_regions.values():   # face_regions默认只有左边的坐标
        left_group: list[int] = []
        right_group: list[int] = []
        for left_point in left_points:
            point_id = int(left_point)
            right_point = sym_map.get(point_id)
            if point_id not in seen:            # 如果左边的点还没有被处理过，就把它加入左边的分组，并标记为已处理
                left_group.append(point_id)
                seen.add(point_id)
            if right_point is not None and right_point not in seen:
                right_group.append(right_point)
                seen.add(right_point)
        ordered.extend(left_group)
        ordered.extend(right_group)
    return tuple(ordered)


# 根据用户指定的区域名称过滤face_regions字典，返回一个新的字典，如果用户指定了不存在的区域名称，则抛出异常
def _filter_face_regions(
    face_regions: dict[str, list[int]],
    *,
    region_names: list[str] | None,
) -> dict[str, list[int]]:
    if region_names is None:
        return face_regions
    filtered: dict[str, list[int]] = {}
    missing: list[str] = []
    for region_name in region_names:
        if region_name not in face_regions:
            missing.append(region_name)
            continue
        filtered[region_name] = face_regions[region_name]
    if missing:
        raise ValueError(f"Unknown face region names: {missing}")
    return filtered


# 根据用户指定的subset_layout和相关配置，返回一个包含所选点ID的元组
# mouth这里似乎不够
def resolve_subset_layout(
    *,
    subset_layout: str,
    subset_layout_source: str | Path,
    subset_layout_extractor_name: str = "mediapipe",
    subset_layout_region_names: list[str] | None = None,
) -> tuple[int, ...]:
    face_regions, symmetric_pairs = load_face_region_config(
        subset_layout_source,
        extractor_name=subset_layout_extractor_name,
    )
    face_regions = _filter_face_regions(
        face_regions,
        region_names=subset_layout_region_names,
    )
    if subset_layout == "face_regions_grouped":
        return build_grouped_face_region_subset(face_regions, symmetric_pairs)
    if subset_layout == "mouth":
        mouth = face_regions.get("mouth")
        if not mouth:
            raise ValueError(f"Extractor config {subset_layout_source} does not define a non-empty mouth region")
        return tuple(int(point_id) for point_id in mouth)
    raise ValueError(f"Unsupported subset_layout: {subset_layout!r}")
