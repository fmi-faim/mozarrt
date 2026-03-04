from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from cyclopts.types import ExistingDirectory
from loguru import logger
from mobiedantic import Dataset, Project, Source
from mobiedantic.generated import SegmentationDisplay, SegmentationDisplay1
from natsort import natsorted
from ngio import open_ome_zarr_container
from scipy.ndimage import center_of_mass, find_objects

if TYPE_CHECKING:
    from ngio import Image, OmeZarrContainer


def _source_path_payload(
    *,
    source_path: Path,
    dataset_path: Path,
    channel_index: int | None = None,
) -> dict[str, int | str]:
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


def _normalize_relative_paths(value):
    if isinstance(value, dict):
        normalized = {}
        for key, item in value.items():
            if key == "relativePath" and isinstance(item, str):
                normalized[key] = item.replace("\\", "/")
            else:
                normalized[key] = _normalize_relative_paths(item)
        return normalized
    if isinstance(value, list):
        return [_normalize_relative_paths(item) for item in value]
    return value


def _write_segmentation_table(
    *,
    dataset_path: Path,
    source_name: str,
    label,
) -> Path:
    arr = label.get_as_numpy()
    axes = label.axes  # e.g. ('y', 'x') or ('z', 'y', 'x')
    is_3d = "z" in axes

    scale = label.pixel_size
    # axis order in array: (y, x) for 2D, (z, y, x) for 3D
    if is_3d:
        scale_factors = [scale.z, scale.y, scale.x]
    else:
        scale_factors = [scale.y, scale.x]

    label_ids = [int(v) for v in np.unique(arr) if int(v) > 0]

    slices = find_objects(arr)
    centroids = center_of_mass(arr, arr, label_ids)
    if len(label_ids) == 1:
        centroids = [centroids]

    rows = []
    for label_id, centroid in zip(label_ids, centroids):
        sl = slices[label_id - 1]  # find_objects is 1-indexed
        if is_3d:
            cz, cy, cx = [float(centroid[i]) * scale_factors[i] for i in range(3)]
            row = {
                "label_id": label_id,
                "anchor_x": cx,
                "anchor_y": cy,
                "anchor_z": cz,
                "bb_min_x": sl[2].start * scale.x,
                "bb_min_y": sl[1].start * scale.y,
                "bb_min_z": sl[0].start * scale.z,
                "bb_max_x": (sl[2].stop - 1) * scale.x,
                "bb_max_y": (sl[1].stop - 1) * scale.y,
                "bb_max_z": (sl[0].stop - 1) * scale.z,
            }
        else:
            cy, cx = [float(centroid[i]) * scale_factors[i] for i in range(2)]
            row = {
                "label_id": label_id,
                "anchor_x": cx,
                "anchor_y": cy,
                "bb_min_x": sl[1].start * scale.x,
                "bb_min_y": sl[0].start * scale.y,
                "bb_max_x": (sl[1].stop - 1) * scale.x,
                "bb_max_y": (sl[0].stop - 1) * scale.y,
            }
        rows.append(row)

    table_dir = dataset_path / "tables" / source_name
    table_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(table_dir / "default.tsv", sep="\t", index=False)
    return table_dir


def _add_segmentation_source(
    *,
    dataset: Dataset,
    source_name: str,
    source_path: Path,
    table_dir: Path,
) -> None:
    image_data_payload = _source_path_payload(
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


def project(
    input_directory: ExistingDirectory,
    output_directory: ExistingDirectory,
    /,
    *,
    description: str | None = None,
):
    project: Project = Project(output_directory)
    project.initialize_model(
        description=description or "Folder Project",
    )
    dataset: Dataset = project.new_dataset(
        name=input_directory.name,
    )

    # Create sources for each zarr (and each channel within it)
    # then add merged grid views for each channel

    zarr_dirs = natsorted(Path(input_directory).glob("*.zarr"))
    if not zarr_dirs:
        raise ValueError(f"No .zarr directories found in '{input_directory}'.")

    first_zarr: OmeZarrContainer = open_ome_zarr_container(zarr_dirs[0])
    first_zarr_image: Image = first_zarr.get_image()

    # Determine if data is 2D based on number of spatial axes (excluding channel dimension)
    axes_names = first_zarr_image.axes
    # Count spatial axes (z, y, x) - typically 2D has [y, x] and 3D has [z, y, x]
    is2d = "z" not in axes_names
    logger.info(f"Axes: {axes_names}, is2D: {is2d}")

    dataset.initialize_with_paths(
        path_dict={},
        is2d=is2d,
    )

    channel_names = [
        channel.label for channel in first_zarr_image.channels_meta.channels
    ]
    logger.info(f"Channel names: {channel_names}")

    # format: sources[channel_name] = pathdict (name: Path)
    sources = defaultdict(dict)
    position_map = defaultdict(list)
    segmentation_source_names: list[str] = []

    for zarr_dir in zarr_dirs:
        logger.info(f"Processing {zarr_dir}")
        position: OmeZarrContainer = open_ome_zarr_container(zarr_dir)
        logger.info(position.meta)
        for channel_name in channel_names:
            logger.info(f"  Channel: {channel_name}")
            source_path = zarr_dir
            source_name = f"{zarr_dir.stem}_{channel_name}"
            sources[channel_name][source_name] = source_path
            position_map[zarr_dir.stem].append(source_name)

        for label_name in position.list_labels():
            logger.info(f"  Label: {label_name}")
            label = position.get_label(label_name)
            source_name = f"{zarr_dir.stem}_{label_name}"
            table_dir = _write_segmentation_table(
                dataset_path=dataset.path,
                source_name=source_name,
                label=label,
            )
            _add_segmentation_source(
                dataset=dataset,
                source_name=source_name,
                source_path=zarr_dir / "labels" / label_name,
                table_dir=table_dir,
            )
            segmentation_source_names.append(source_name)

    for channel_index, channel_name in enumerate(channel_names):
        logger.info(f"Adding source for channel {channel_name}...")
        dataset.add_sources(
            path_dict=sources[channel_name],
            channel_index=channel_index,
            data_format="ome.zarr",
        )
        channel_sources = list(sources[channel_name])
        if len(channel_sources) > 1:
            dataset.add_merged_grid(
                name=f"merged_grid_{channel_name}",
                sources=channel_sources,
            )

    # Add region view containing all positions
    if dataset.model.views["default"].sourceDisplays is None:
        dataset.model.views["default"].sourceDisplays = []

    dataset.add_region_view(
        name="all_positions",
        map_of_sources=position_map,
    )

    for segmentation_source_name in segmentation_source_names:
        dataset.model.views["default"].sourceDisplays.append(
            SegmentationDisplay(
                segmentationDisplay=SegmentationDisplay1(
                    name=segmentation_source_name,
                    sources=[segmentation_source_name],
                    opacity=0.5,
                    lut="glasbey",
                )
            )
        )

    normalized_sources = {}
    for source_name, source in dataset.model.sources.items():
        source_data = source.model_dump(exclude_none=True, by_alias=True)
        normalized_sources[source_name] = Source(
            **_normalize_relative_paths(source_data)
        )
    dataset.model.sources = normalized_sources

    dataset.save()
    project.save()
