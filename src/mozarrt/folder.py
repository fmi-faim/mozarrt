from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from cyclopts.types import ExistingDirectory
from loguru import logger
from mobiedantic import Dataset, Project, Source
from natsort import natsorted
from ngio import open_ome_zarr_container

if TYPE_CHECKING:
    from ngio import Image, OmeZarrContainer


def _source_path_payload(
    *,
    source_path: Path,
    dataset_path: Path,
    channel_index: int | None = None,
) -> dict[str, int | str]:
    try:
        payload: dict[str, int | str] = {
            "relativePath": str(
                Path(source_path).relative_to(dataset_path, walk_up=True)
            ),
        }
    except (ValueError, TypeError):
        payload = {
            "absolutePath": str(Path(source_path).absolute()),
        }

    if channel_index is not None:
        payload["channel"] = channel_index
    return payload


def _extract_observed_label_ids(label) -> list[int]:
    label_values = np.unique(label.get_as_numpy())
    return [int(value) for value in label_values if int(value) > 0]


def _write_segmentation_table(
    *,
    dataset_path: Path,
    source_name: str,
    label_ids: list[int],
) -> Path:
    table_dir = dataset_path / "tables" / source_name
    table_dir.mkdir(parents=True, exist_ok=True)

    table = pd.DataFrame(
        {
            "label_id": label_ids,
            "annotation": ["" for _ in label_ids],
        }
    )
    table.to_csv(table_dir / "default.tsv", sep="\t", index=False)
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
    table_relative_path = str(table_dir.relative_to(dataset.path))
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
            label_ids = _extract_observed_label_ids(label)
            source_name = f"{zarr_dir.stem}_{label_name}"
            table_dir = _write_segmentation_table(
                dataset_path=dataset.path,
                source_name=source_name,
                label_ids=label_ids,
            )
            _add_segmentation_source(
                dataset=dataset,
                source_name=source_name,
                source_path=zarr_dir,
                table_dir=table_dir,
            )
            position_map[zarr_dir.stem].append(source_name)

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

    dataset.save()
    project.save()
