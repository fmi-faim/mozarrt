from collections import defaultdict
from pathlib import Path

from cyclopts.types import ExistingDirectory
from loguru import logger
from mobiedantic import Dataset, Project, Source
from mobiedantic.generated import (
    ImageDisplay,
    ImageDisplay1,
    MergedGrid,
    MergedGrid1,
    SegmentationDisplay,
    SegmentationDisplay1,
    TransformedGrid,
    TransformedGrid1,
)
from ngio import open_ome_zarr_container, open_ome_zarr_plate
from ngio.hcs import OmeZarrPlate

from mozarrt._table_utils import (
    add_segmentation_source,
    compute_label_rows,
    normalize_relative_paths,
    write_segmentation_table,
)
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
    output_directory: Path,
    /,
) -> Project:
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
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

    # From the first existing well, get subpaths and channel names.
    # Some plates have sparse occupancy (e.g. missing C/03), so do not assume
    # that the first row/column combination exists as a concrete well.
    wells_paths = list(plate.wells_paths())
    if not wells_paths:
        raise ValueError(
            f"No wells found in plate '{plate_zarr_path}'. Cannot create project."
        )

    # Then, for each subpath and each channel, create sources and merged grid views
    first_well_path = wells_paths[0]
    first_row, first_col = first_well_path.split("/")
    first_well = plate.get_well(row=first_row, column=first_col)
    path_names = first_well.paths()
    logger.info(f"Path names in first well: {path_names}")
    first_well_image = first_well.get_image(path_names[0]).get_image()
    channel_names = [
        channel.label for channel in first_well_image.channels_meta.channels
    ]
    logger.info(f"Channel names: {channel_names}")

    # Read per-channel contrast limits from OME-Zarr metadata of the first well
    contrast_limits: dict[str, list[float]] = {}
    for ch in first_well_image.channels_meta.channels:
        v = ch.channel_visualisation
        contrast_limits[ch.label] = [v.start or 0.0, v.end or 65535.0]
    logger.info(f"Contrast limits: {contrast_limits}")

    # sources_path[sub_path][channel_name] = dict[source_name, source_path]  (for add_sources)
    # sources_pos[sub_path][channel_name]  = list[(source_name, col_idx, row_idx)]  (for add_merged_grid)
    sources_path = defaultdict(lambda: defaultdict(dict))
    sources_pos: dict = defaultdict(lambda: defaultdict(list))
    sources_per_well = defaultdict(list)
    # label_rows[label_name] = accumulated list of row dicts for combined (analysis) table
    # label_source_names[label_name] = list of (source_name, label_path, col_idx, row_idx)
    label_rows: dict[str, list[dict]] = defaultdict(list)
    label_rows_by_source: dict[str, dict[str, list[dict]]] = defaultdict(dict)
    label_source_names: dict[str, list[tuple]] = defaultdict(list)

    for well_path in wells_paths:
        logger.info(f"Processing well: {well_path}")
        row_letter, col_number = well_path.split("/")
        row_idx = plate.rows.index(row_letter)
        col_idx = plate.columns.index(col_number)
        # add sources for each subpath and channel
        for sub_path in path_names:
            for channel in channel_names:
                logger.info(f"  Channel: {channel}")
                source_path = plate_zarr_path / well_path / sub_path
                source_name = f"{well_path.replace('/', '')}_{channel}"
                sources_path[sub_path][channel][source_name] = source_path
                sources_pos[sub_path][channel].append((source_name, col_idx, row_idx))
                sources_per_well[well_path].append(source_name)

        # collect label sources from the first sub_path only
        sub_path_labels = path_names[0]
        try:
            field_container = open_ome_zarr_container(
                plate_zarr_path / well_path / sub_path_labels
            )
            for label_name in field_container.list_labels():
                logger.info(f"  Label: {label_name}")
                label = field_container.get_label(label_name)
                seg_source_name = f"{well_path.replace('/', '')}_{label_name}"
                # BDV/MoBIE computes source extent as (N-1)*scale (center of
                # first pixel to center of last pixel). Use the same formula so
                # table anchor offsets stay in sync with TransformedGrid slot
                # positions. Using N*scale instead causes 1-pixel drift per
                # column/row that accumulates across the plate.
                phys_w = (label.shape[-1] - 1) * label.pixel_size.x
                phys_h = (label.shape[-2] - 1) * label.pixel_size.y
                # Per-well table: LOCAL coordinates (no offset).
                # MoBIE applies the TransformedGrid translation automatically
                # during navigation – adding offset here would double it and
                # cause the viewer to jump too far right/down for all wells
                # after the first.
                local_rows = compute_label_rows(
                    label,
                    well=well_path,
                    plate_name=plate_zarr_path.name,
                )
                if not local_rows:
                    logger.warning(
                        f"  Label '{label_name}' in {well_path} has no objects; skipping source."
                    )
                    continue
                # Combined analysis table: GLOBAL coordinates (offset added)
                # so positions are comparable across the full plate.
                offset_x = col_idx * phys_w
                offset_y = row_idx * phys_h
                global_rows = compute_label_rows(
                    label,
                    label_image_id=seg_source_name,
                    offset_x=offset_x,
                    offset_y=offset_y,
                    well=well_path,
                    plate_name=plate_zarr_path.name,
                )
                label_rows[label_name].extend(global_rows)
                label_rows_by_source[label_name][seg_source_name] = global_rows
                label_path = (
                    plate_zarr_path
                    / well_path
                    / sub_path_labels
                    / "labels"
                    / label_name
                )
                label_source_names[label_name].append(
                    (seg_source_name, label_path, col_idx, row_idx, phys_w, phys_h)
                )
        except Exception as exc:
            logger.warning(f"  Could not read labels for {well_path}: {exc}")

    for sub_path, channel_dict in sources_path.items():
        for channel_index, channel_name in enumerate(channel_names):
            logger.info(
                f"Adding source for subpath {sub_path}, channel {channel_name}..."
            )
            plate_dataset.add_sources(
                path_dict=channel_dict[channel_name],
                channel_index=channel_index,
                data_format="ome.zarr",
            )
            if plate_dataset.model.views["default"].sourceTransforms is None:
                plate_dataset.model.views["default"].sourceTransforms = []
            if plate_dataset.model.views["default"].sourceDisplays is None:
                plate_dataset.model.views["default"].sourceDisplays = []
            pos_list = sources_pos[sub_path][channel_name]
            merged_name = f"merged_grid_{sub_path}_{channel_name}"
            # Use margin=0 so grid slots sit at exactly col*phys_w, row*phys_h –
            # matching the table anchor coordinates and the label TransformedGrid.
            plate_dataset.model.views["default"].sourceTransforms.append(
                MergedGrid(
                    mergedGrid=MergedGrid1(
                        sources=[sn for sn, _, _ in pos_list],
                        positions=[(c, r) for _, c, r in pos_list],
                        mergedGridSourceName=merged_name,
                        margin=0.0,
                    )
                )
            )
            cl = contrast_limits.get(channel_name, [0.0, 65535.0])
            plate_dataset.model.views["default"].sourceDisplays.append(
                ImageDisplay(
                    imageDisplay=ImageDisplay1(
                        name=merged_name,
                        color="white",
                        opacity=1.0,
                        contrastLimits=cl,
                        sources=[merged_name],
                    )
                )
            )

    # Add region view containing all wells
    plate_dataset.add_region_view(
        name="all_wells",
        map_of_sources=sources_per_well,
    )

    # Add label segmentation sources (shared combined table per label name).
    # A TransformedGrid sourceTransform (margin=0) positions each per-well source
    # at its correct grid slot using the same formula as the intensity MergedGrid.
    # Unlike MergedGrid, TransformedGrid keeps each source independent so label
    # IDs (all starting at 1 per well) never collide.
    for label_name, source_entries in label_source_names.items():
        combined_table_dir = plate_dataset.path / "tables" / label_name
        if label_rows[label_name]:
            # Canonical combined table for analysis/export.
            # Stored at tables/<label>/default.tsv (higher-level single table).
            write_segmentation_table(label_rows[label_name], combined_table_dir)

        if len(source_entries) < 2:
            logger.warning(
                f"Skipping label view '{label_name}': need at least 2 non-empty wells for TransformedGrid, got {len(source_entries)}."
            )
            continue

        if plate_dataset.model.views["default"].sourceTransforms is None:
            plate_dataset.model.views["default"].sourceTransforms = []
        if plate_dataset.model.views["default"].sourceDisplays is None:
            plate_dataset.model.views["default"].sourceDisplays = []

        all_seg_source_names: list[str] = []
        nested_sources: list[list[str]] = []
        grid_positions: list[tuple[int, int]] = []

        for (
            seg_source_name,
            label_path,
            col_idx,
            row_idx,
            _phys_w,
            _phys_h,
        ) in source_entries:
            source_table_dir = combined_table_dir / seg_source_name
            write_segmentation_table(
                label_rows_by_source[label_name][seg_source_name],
                source_table_dir,
            )
            add_segmentation_source(
                dataset=plate_dataset,
                source_name=seg_source_name,
                source_path=label_path,
                table_dir=source_table_dir,
            )
            all_seg_source_names.append(seg_source_name)
            nested_sources.append([seg_source_name])
            grid_positions.append((col_idx, row_idx))

        plate_dataset.model.views["default"].sourceTransforms.append(
            TransformedGrid(
                transformedGrid=TransformedGrid1(
                    nestedSources=nested_sources,
                    positions=grid_positions,
                    margin=0.0,
                )
            )
        )
        plate_dataset.model.views["default"].sourceDisplays.append(
            SegmentationDisplay(
                segmentationDisplay=SegmentationDisplay1(
                    name=label_name,
                    sources=all_seg_source_names,
                    opacity=0.5,
                    lut="glasbey",
                )
            )
        )

    # Normalise path separators
    normalized_sources = {}
    for sname, source in plate_dataset.model.sources.items():
        source_data = source.model_dump(exclude_none=True, by_alias=True)
        normalized_sources[sname] = Source(**normalize_relative_paths(source_data))
    plate_dataset.model.sources = normalized_sources

    plate_dataset.save()
    project.save()
    return project
