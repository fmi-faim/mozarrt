from cyclopts.types import ExistingDirectory
from loguru import logger
from ngio import open_ome_zarr_container, open_ome_zarr_plate
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
