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

    for channel_index, channel_name in enumerate(channel_names):
        logger.info(f"Adding source for channel {channel_name}...")
        dataset.add_sources(
            path_dict=sources[channel_name],
            channel_index=channel_index,
            data_format="ome.zarr",
        )
        dataset.add_merged_grid(
            name=f"merged_grid_{channel_name}",
            sources=list(sources[channel_name]),
        )
    
    # Add region view containing all positions
    dataset.add_region_view(
        name="all_positions",
        map_of_sources=position_map,
    )

    dataset.save()
    project.save()
