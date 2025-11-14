from collections import defaultdict
from cyclopts.types import ExistingDirectory
from loguru import logger
from natsort import natsorted
from ngio import OmeZarrContainer, open_ome_zarr_container, open_ome_zarr_plate
from ngio.hcs import OmeZarrPlate
from mobiedantic import Project, Dataset
from pathlib import Path

from mozarrt.mobie_collection import MoBIECollectionEntry, collection_dataframe

def _create_intensities_entry(
        uri: str,
        name: str,
        channel: int,
        grid: str,
        display: str,
        # grid_position: tuple[int, int],
    ) -> MoBIECollectionEntry:
    return MoBIECollectionEntry(
        uri=uri,
        name=name,
        type="intensities",
        format="OmeZarr",
        channel=channel,
        contrast_limits=(0, 255),
        view="all",
        display=display,
        grid=grid,
        # grid_position=grid_position,
    )

def _create_labels_entry(
        uri: str,
        name: str,
        channel: int,
        grid: str,
        display: str,
        # grid_position: tuple[int, int],
    ) -> MoBIECollectionEntry:
    return MoBIECollectionEntry(
        uri=uri,
        name=name,
        type="labels",
        format="OmeZarr",
        channel=channel,
        view="all",
        grid=grid,
        display=display,
        # grid_position=grid_position,
    )

def _create_spots_entry(
        uri: str,
        name: str,
        grid: str,
        display: str,
        # grid_position: tuple[int, int],
    ) -> MoBIECollectionEntry:
    return MoBIECollectionEntry(
        uri=uri,
        name=name,
        type="spots",
        view="all",
        grid=grid,
        display=display,
        # grid_position=grid_position,
    )

def plate(
    plate_zarr_path: ExistingDirectory,
    output_directory: ExistingDirectory,
    /,
):
    # Open plate zarr container
    plate = open_ome_zarr_plate(plate_zarr_path)

    mobie_collection_entries: list[MoBIECollectionEntry] = []

    sub_path = "0"

    for well_path, well in plate.get_wells().items():
        logger.info(f"Processing well: {well_path}")
        assert sub_path in well.paths()
        # For each well, generate MoBIECollectionEntry
        well_image = well.get_image(sub_path).get_image()
        logger.info(well_image.channels_meta)
        for ch_idx, channel in enumerate(well_image.channels_meta.channels):
            logger.info(f"  Channel: {channel.label}")
            entry = _create_intensities_entry(
                uri=f"{plate_zarr_path}/{well_path}/{sub_path}",
                name=f"{well_path}_c{ch_idx}",
                channel=ch_idx,
                display=channel.label,
                grid=f"{ch_idx}_grid",
                # grid_position=(0, 0), # TODO compute position from well_path
            )
            mobie_collection_entries.append(entry)

    df = collection_dataframe(mobie_collection_entries)
    output_path = Path(output_directory) / "mobie_collection.csv"
    df.to_csv(output_path, index=False)

def folder(
    input_directory: ExistingDirectory,
    output_directory: ExistingDirectory,
    /,
    *,
    intensity_paths: list[str] | None = None,
    label_paths: list[str] | None = None,
):
    mobie_collection_entries: list[MoBIECollectionEntry] = []
    # Loop over all (.zarr) directories in input_directory
    for zarr_dir in Path(input_directory).rglob("*.zarr"):
        # Open zarr container
        container = open_ome_zarr_container(zarr_dir)
        logger.info(f"Processing {zarr_dir}")
        image = container.get_image()
        logger.info(image)
        logger.info(image.path)
        logger.info(image.channels_meta)
        for channel in image.channel_labels:
            logger.info(channel)
            logger.info(image.get_channel_idx(channel))
        for channel in image.channels_meta.channels:
            logger.info(channel)
            logger.info(channel.label)
            logger.info(channel.channel_visualisation.start)
        # Append an intensities entry for each channel
    #     for channel_index in range(image.get_channel_count()):
    #         entry = _create_intensities_entry(
    #             uri=str(zarr_dir.resolve()), # TODO path to image
    #             name=f"{zarr_dir.stem}_c{channel_index}",
    #             channel=channel_index,
    #             grid=f"channel_{channel_index}_grid",
    #             grid_position=(0, 0), # TODO compute position from filename/folder content
    #         )
    #         mobie_collection_entries.append(entry)

    # df = collection_dataframe(mobie_collection_entries)
    # output_path = output_directory / "mobie_collection.csv"
    # df.to_csv(output_path, index=False)

def create_plate_project(
    plate_zarr_path: ExistingDirectory,
    output_directory: ExistingDirectory,
    /,
) -> Project:
    plate: OmeZarrPlate = open_ome_zarr_plate(plate_zarr_path)
    logger.info(f"Creating MoBIE Project for plate at {plate_zarr_path}")
    # Initialize Project
    project: Project = Project(output_directory)
    project.initialize_model(
        description="Test project",
    )

    # Create Dataset for plate
    plate_dataset: Dataset = project.new_dataset(
        # name=plate.meta.plate.name,
        name=plate_zarr_path.name,
    )
    plate_dataset.initialize_with_paths(
        path_dict={},
        is2d=True,
    )

    plate_dataset.save()

    # From the first well, get subpaths and channel names
    # Then, for each subpath and each channel, create sources and merged grid views
    first_well = plate.get_well(row=plate.rows[0], column=plate.columns[0])
    path_names = first_well.paths()
    logger.info(f"Path names in first well: {path_names}")
    first_well_image = first_well.get_image(path_names[0]).get_image()
    channel_names = [
        channel.label for channel in first_well_image.channels_meta.channels
    ]
    logger.info(f"Channel names: {channel_names}")

    # format: sources[sub_path][channel_name] = pathdict (name: Path)
    sources = defaultdict(lambda: defaultdict(dict))
    sources_per_well = defaultdict(list)

    for well_path in plate.wells_paths():
        logger.info(f"Processing well: {well_path}")
        # add sources for each subpath and channel
        for sub_path in path_names:
            for channel in channel_names:
                logger.info(f"  Channel: {channel}")
                source_path = plate_zarr_path / well_path / sub_path
                source_name = f"{well_path.replace("/", "")}_{channel}"
                sources[sub_path][channel][source_name] = source_path
                sources_per_well[well_path].append(source_name)
    
    for sub_path, channel_dict in sources.items():
        for channel_index, channel_name in enumerate(channel_names):
            logger.info(f"Adding source for subpath {sub_path}, channel {channel_name}...")
            plate_dataset.add_sources(
                path_dict=channel_dict[channel_name],
                channel_index=channel_index,
                data_format="ome.zarr",
            )
            plate_dataset.add_merged_grid(
                name=f"merged_grid_{sub_path}_{channel_name}",
                sources=list(channel_dict[channel_name]),
            )
    
    # Add region view containing all wells
    plate_dataset.add_region_view(
        name="all_wells",
        map_of_sources=sources_per_well,
    )


    plate_dataset.save()
    project.save()
    # add sources for each well and channel
    # add merged grid view for each channel, containing all well sources for that channel
    # add sources for labels

