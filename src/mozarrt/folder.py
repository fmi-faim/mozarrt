from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

from cyclopts.types import ExistingDirectory
from loguru import logger
from mobiedantic import Dataset, Project, Source
from mobiedantic.generated import SegmentationDisplay, SegmentationDisplay1
from natsort import natsorted
from ngio import open_ome_zarr_container

from mozarrt._table_utils import (
    add_segmentation_source,
    compute_label_rows,
    normalize_relative_paths,
    write_segmentation_table,
)

if TYPE_CHECKING:
    from ngio import Image, OmeZarrContainer


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
            rows = compute_label_rows(label)
            table_dir = write_segmentation_table(
                rows, dataset.path / "tables" / source_name
            )
            add_segmentation_source(
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
            **normalize_relative_paths(source_data)
        )
    dataset.model.sources = normalized_sources

    dataset.save()
    project.save()
