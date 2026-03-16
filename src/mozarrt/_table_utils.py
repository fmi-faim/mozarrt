"""Shared utilities for computing MoBIE segmentation tables and sources."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from mobiedantic import Dataset, Source
from scipy.ndimage import center_of_mass, find_objects


def source_path_payload(
    *,
    source_path: Path,
    dataset_path: Path,
    channel_index: int | None = None,
) -> dict[str, int | str]:
    """Return imageData payload dict with relative or absolute path."""
    try:
        relative_path = Path(source_path).relative_to(dataset_path, walk_up=True)
        payload: dict[str, int | str] = {
            "relativePath": relative_path.as_posix(),
        }
    except (ValueError, TypeError):
        payload = {
            "absolutePath": str(Path(source_path).absolute()),
        }

    if channel_index is not None:
        payload["channel"] = channel_index
    return payload


def normalize_relative_paths(value: Any) -> Any:
    """Recursively replace backslashes with forward slashes in relativePath values."""
    if isinstance(value, dict):
        normalized = {}
        for key, item in value.items():
            if key == "relativePath" and isinstance(item, str):
                normalized[key] = item.replace("\\", "/")
            else:
                normalized[key] = normalize_relative_paths(item)
        return normalized
    if isinstance(value, list):
        return [normalize_relative_paths(item) for item in value]
    return value


def compute_label_rows(
    label,
    *,
    label_image_id: str | None = None,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
    offset_z: float = 0.0,
    well: str | None = None,
    plate_name: str | None = None,
) -> list[dict]:
    """Compute per-label rows with anchors and bounding boxes.

    Parameters
    ----------
    label:
        An ngio label object (supports .get_as_numpy(), .axes, .pixel_size).
    label_image_id:
        If given, add a ``label_image_id`` column so multiple sources can share
        one combined table.
    offset_x, offset_y, offset_z:
        Physical-unit offsets to add to all spatial coordinates (used when
        placing individual fields into a grid, e.g. HCS plates).
    well:
        Well path string (e.g. ``"C/03"``), added as a ``well`` column.
    plate_name:
        Plate name (e.g. ``"exp164-diff0"``), added as a ``plate_name`` column.

    Returns
    -------
    list[dict]
        One dict per label ID.
    """
    arr = label.get_as_numpy()
    axes = label.axes  # e.g. ('y', 'x') or ('z', 'y', 'x')
    is_3d = "z" in axes

    scale = label.pixel_size
    if is_3d:
        scale_factors = [scale.z, scale.y, scale.x]
    else:
        scale_factors = [scale.y, scale.x]

    label_ids = [int(v) for v in np.unique(arr) if int(v) > 0]
    if not label_ids:
        return []

    slices = find_objects(arr)
    centroids = center_of_mass(arr, arr, label_ids)
    # scipy returns a list of tuples when index is a list (even length-1 lists);
    # no special-casing needed – guard only against scalar return for safety
    if not isinstance(centroids, list):
        centroids = [centroids]

    rows = []
    for label_id, centroid in zip(label_ids, centroids):
        sl = slices[label_id - 1]  # find_objects is 1-indexed
        if is_3d:
            cz, cy, cx = [float(centroid[i]) * scale_factors[i] for i in range(3)]
            row = {
                "label_id": label_id,
                "anchor_x": cx + offset_x,
                "anchor_y": cy + offset_y,
                "anchor_z": cz + offset_z,
                "bb_min_x": sl[2].start * scale.x + offset_x,
                "bb_min_y": sl[1].start * scale.y + offset_y,
                "bb_min_z": sl[0].start * scale.z + offset_z,
                "bb_max_x": (sl[2].stop - 1) * scale.x + offset_x,
                "bb_max_y": (sl[1].stop - 1) * scale.y + offset_y,
                "bb_max_z": (sl[0].stop - 1) * scale.z + offset_z,
            }
        else:
            cy, cx = [float(centroid[i]) * scale_factors[i] for i in range(2)]
            row = {
                "label_id": label_id,
                "anchor_x": cx + offset_x,
                "anchor_y": cy + offset_y,
                "bb_min_x": sl[1].start * scale.x + offset_x,
                "bb_min_y": sl[0].start * scale.y + offset_y,
                "bb_max_x": (sl[1].stop - 1) * scale.x + offset_x,
                "bb_max_y": (sl[0].stop - 1) * scale.y + offset_y,
            }
        if label_image_id is not None:
            row["label_image_id"] = label_image_id
        if well is not None:
            row["well"] = well
        if plate_name is not None:
            row["plate_name"] = plate_name
        rows.append(row)
    return rows


def write_segmentation_table(rows: list[dict], table_dir: Path) -> Path:
    """Write *rows* to ``table_dir/default.tsv``.  Returns *table_dir*."""
    table_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    priority = [c for c in ("label_id", "well", "plate_name") if c in df.columns]
    rest = [c for c in df.columns if c not in priority]
    df = df[priority + rest]
    df.to_csv(table_dir / "default.tsv", sep="\t", index=False)
    return table_dir
    return table_dir


def add_segmentation_source(
    *,
    dataset: Dataset,
    source_name: str,
    source_path: Path,
    table_dir: Path,
) -> None:
    """Register a segmentation source in *dataset* pointing to *table_dir*."""
    image_data_payload = source_path_payload(
        source_path=source_path,
        dataset_path=dataset.path,
        channel_index=None,
    )
    table_relative_path = table_dir.relative_to(dataset.path).as_posix()
    source_data = {
        "segmentation": {
            "imageData": {
                "ome.zarr": image_data_payload,
            },
            "tableData": {
                "tsv": {
                    "relativePath": table_relative_path,
                }
            },
        }
    }
    dataset.model.sources[source_name] = Source(**source_data)
