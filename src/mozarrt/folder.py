from collections import defaultdict
from cyclopts.types import ExistingDirectory
from loguru import logger
from mobiedantic import Project, Dataset
from natsort import natsorted
from ngio import open_ome_zarr_container
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ngio import OmeZarrContainer, Image


def _hex_rgb_to_rgba(color_hex: str | None) -> str | None:
    if not color_hex:
        return None
    normalized = color_hex.lstrip("#")
    if len(normalized) != 6:
        return None
    try:
        red = int(normalized[0:2], 16)
        green = int(normalized[2:4], 16)
        blue = int(normalized[4:6], 16)
    except ValueError:
        return None
    return f"{red}-{green}-{blue}-255"


def _channel_visualisation(channel_meta):
    return getattr(
        channel_meta,
        "channel_visualization",
        getattr(channel_meta, "channel_visualisation", None),
    )


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
    channel_colors: dict[str, str] = {}
    channel_contrast_limits = {
        channel_name: [float("inf"), float("-inf")] for channel_name in channel_names
    }

    for zarr_dir in zarr_dirs:
        logger.info(f"Processing {zarr_dir}")
        position: OmeZarrContainer = open_ome_zarr_container(zarr_dir)
        logger.info(position.meta)
        image = position.get_image()
        channels_meta = getattr(position.meta, "channel_meta", None) or getattr(
            position.meta, "channels_meta", None
        )
        for channel_index, channel_name in enumerate(channel_names):
            logger.info(f"  Channel: {channel_name}")
            source_path = zarr_dir
            source_name = f"{zarr_dir.stem}_{channel_name}"
            sources[channel_name][source_name] = source_path
            position_map[zarr_dir.stem].append(source_name)
            if (
                channel_name not in channel_colors
                and channels_meta is not None
                and len(channels_meta.channels) > channel_index
            ):
                channel_vis = _channel_visualisation(channels_meta.channels[channel_index])
                if channel_vis is not None:
                    color = _hex_rgb_to_rgba(channel_vis.color)
                    if color is not None:
                        channel_colors[channel_name] = color
            channel_vis = _channel_visualisation(
                image.meta.channels_meta.channels[channel_index]
            )
            if channel_vis is not None:
                channel_contrast_limits[channel_name][0] = min(
                    channel_contrast_limits[channel_name][0], channel_vis.start
                )
                channel_contrast_limits[channel_name][1] = max(
                    channel_contrast_limits[channel_name][1], channel_vis.end
                )

    for channel_index, channel_name in enumerate(channel_names):
        logger.info(f"Adding source for channel {channel_name}...")
        dataset.add_image_sources(
            path_dict=sources[channel_name],
            channel_index=channel_index,
            data_format="ome.zarr",
        )
        merged_grid_name = f"merged_grid_{channel_name}"
        dataset.add_merged_grid(
            name=merged_grid_name,
            sources=list(sources[channel_name]),
        )
        contrast_limits = channel_contrast_limits[channel_name]
        if contrast_limits[0] == float("inf"):
            contrast_limits = [0.0, 255.0]
        dataset.add_image_display(
            name=merged_grid_name,
            sources=[merged_grid_name],
            color=channel_colors.get(channel_name, "white"),
            contrast_limits=(contrast_limits[0], contrast_limits[1]),
        )

    # Add region view containing all positions
    dataset.add_region_display(
        name="all_positions",
        map_of_sources=position_map,
    )

    dataset.save()
    project.save()
