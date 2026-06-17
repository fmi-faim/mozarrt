from collections import defaultdict
from cyclopts.types import ExistingDirectory
import pandas as pd
from pathlib import Path
import shutil
from mobiedantic import Project, Dataset
from loguru import logger
from skimage.measure import regionprops_table
from tqdm.auto import tqdm

from ngio import open_ome_zarr_plate, OmeZarrPlate
from .utils import update_channel_display_metadata


def project(
    plate_zarr_path: ExistingDirectory,
    output_directory: ExistingDirectory,
    /,
    *,
    exclude_labels: list[str] | None = None,
    force_overwrite_dataset: bool = False,
    include_feature_tables: bool = False,
) -> Project:
    """Create a MoBIE project from an HCS plate OME-Zarr dataset.

    Parameters:
    -----------
    plate_zarr_path: ExistingDirectory
        Path to the HCS plate OME-Zarr dataset.
    output_directory: ExistingDirectory
        Directory where the MoBIE project will be created.
    exclude_labels: list[str] | None
        Labels to exclude from the project. Default is None.
    force_overwrite_dataset: bool
        Whether to force overwrite an existing dataset. Default is False.
    include_feature_tables: bool
        Whether to include existing FeatureTables from the OME-Zarr dataset as
        additional MoBIE table data. Only FeatureTables whose reference label is
        included in the dataset (i.e. not excluded) will be transferred.
        Default is False.
    """
    plate: OmeZarrPlate = open_ome_zarr_plate(plate_zarr_path)
    excluded_label_names = set(exclude_labels or [])
    logger.info(f"Creating MoBIE Project for plate at {plate_zarr_path}")
    output_path = Path(output_directory)
    project_json_path = output_path / "project.json"

    project: Project = Project(output_directory)
    if project_json_path.exists():
        logger.info(f"Loading existing MoBIE project from {output_directory}")
        project.load()
    else:
        logger.info(f"Initializing new MoBIE project at {output_directory}")
        project.initialize_model(
            description="Test project",
        )

    dataset_name = Path(plate_zarr_path).name
    existing_dataset_names = {dataset.root for dataset in project.model.datasets}
    if dataset_name in existing_dataset_names:
        if not force_overwrite_dataset:
            raise ValueError(
                f"Dataset '{dataset_name}' already exists in project "
                f"'{output_directory}'. Re-run with force_overwrite_dataset=True "
                "to overwrite it."
            )
        logger.warning(
            f"Overwriting existing dataset '{dataset_name}' in project "
            f"'{output_directory}'."
        )
        project.model.datasets = [
            dataset
            for dataset in project.model.datasets
            if dataset.root != dataset_name
        ]
        existing_dataset_path = output_path / dataset_name
        if existing_dataset_path.exists():
            shutil.rmtree(existing_dataset_path)

    # Create Dataset for plate
    plate_dataset: Dataset = project.new_dataset(
        # name=plate.meta.plate.name,
        name=dataset_name,
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
    labels = first_well.get_image(path_names[0]).list_labels()
    logger.info(f"Labels: {labels}")

    # format: sources[sub_path][channel_name] = pathdict (name: Path)
    image_sources = defaultdict(lambda: defaultdict(dict))
    image_sources_per_well = defaultdict(list)
    channel_colors = defaultdict(dict)
    channel_contrast_limits = defaultdict(
        lambda: defaultdict(lambda: [float("inf"), float("-inf")])
    )
    image_positions = defaultdict(lambda: defaultdict(list))
    label_sources = defaultdict(lambda: defaultdict(dict))
    label_positions = defaultdict(lambda: defaultdict(list))
    label_sources_per_well = defaultdict(list)
    label_tables = defaultdict(lambda: defaultdict(dict))
    additional_feature_tables = defaultdict(lambda: defaultdict(list))
    well_label_table_dirs: dict[tuple[str, str, str], Path] = {}
    ft_columns_registry: dict[tuple[str, str], dict[str, list]] = {}
    dataset_path = output_path / dataset_name

    # loop through all wells
    for well_path, well_object in tqdm(
        plate.get_wells().items(), desc="Processing wells"
    ):
        logger.info(f"Processing well: {well_path}")
        row_name, column_name = well_path.split("/")
        well_position = (plate.columns.index(column_name), plate.rows.index(row_name))
        for sub_path in well_object.paths():
            logger.info(f"  Processing subpath: {sub_path}")
            image_container = well_object.get_image(sub_path)
            image = image_container.get_image()
            for channel in channel_names:
                logger.info(f"    Channel: {channel}")
                source_path = plate_zarr_path / well_path / sub_path
                source_name = f"{well_path.replace('/', '')}_{sub_path}_{channel}"
                image_sources[sub_path][channel][source_name] = source_path
                image_positions[sub_path][channel].append(well_position)
                image_sources_per_well[well_path].append(source_name)
                update_channel_display_metadata(
                    channel_name=channel,
                    channel_colors=channel_colors[sub_path],
                    channel_contrast_limits=channel_contrast_limits[sub_path],
                    container_channels_meta=image_container.meta.channels_meta,
                    image_channels_meta=image.meta.channels_meta,
                )

            labels = image_container.list_labels()
            for label in labels:
                if label in excluded_label_names:
                    logger.info(f"    Skipping excluded label: {label}")
                    continue
                logger.info(f"    Label name {label}")
                source_path = plate_zarr_path / well_path / sub_path / "labels" / label
                logger.debug(f"    Label source path: {source_path}")
                source_name = f"{well_path.replace('/', '')}_{sub_path}_{label}"
                label_array = image_container.get_label(label).get_array()
                table_df = pd.DataFrame(
                    regionprops_table(
                        label_image=label_array,
                        properties=["label", "centroid", "bbox"],
                        spacing=image_container.get_image().pixel_size.x,
                    )
                )
                table_relative_directory = (
                    Path("tables") / f"{well_path.replace('/', '')}_{label}"
                )
                table_abs_path = dataset_path / table_relative_directory / "default.tsv"
                table_abs_path.parent.mkdir(parents=True, exist_ok=True)
                table_df.to_csv(table_abs_path, sep="\t", index=False)
                label_sources[sub_path][label][source_name] = source_path
                label_positions[sub_path][label].append(well_position)
                label_tables[sub_path][label][source_name] = table_abs_path.parent
                label_sources_per_well[well_path].append(source_name)
                well_label_table_dirs[(well_path, sub_path, label)] = table_abs_path.parent

                if include_feature_tables:
                    feature_table_names = image_container.list_tables(
                        filter_types="feature_table"
                    )
                    for ft_name in feature_table_names:
                        ft = image_container.get_feature_table(ft_name)
                        if ft.reference_label != label:
                            continue
                        logger.info(
                            f"    Including feature table '{ft_name}' "
                            f"for label '{label}'"
                        )
                        ft_df = ft.load_as_pandas_df().reset_index()
                        ft_filename = f"{ft_name}.tsv"
                        ft_abs_path = table_abs_path.parent / ft_filename
                        ft_df.to_csv(ft_abs_path, sep="\t", index=False)
                        key = (sub_path, label)
                        if key not in ft_columns_registry:
                            ft_columns_registry[key] = {}
                        if ft_name not in ft_columns_registry[key]:
                            ft_columns_registry[key][ft_name] = list(ft_df.columns)
                        if ft_filename not in additional_feature_tables[sub_path][label]:
                            additional_feature_tables[sub_path][label].append(
                                ft_filename
                            )

    if include_feature_tables:
        for (w_path, sp, lbl), table_dir in well_label_table_dirs.items():
            for ft_name, columns in ft_columns_registry.get((sp, lbl), {}).items():
                ft_path = table_dir / f"{ft_name}.tsv"
                if not ft_path.exists():
                    logger.info(
                        f"Writing header-only feature table '{ft_name}' "
                        f"for well '{w_path}', label '{lbl}'"
                    )
                    pd.DataFrame(columns=columns).to_csv(ft_path, sep="\t", index=False)

    for sub_path, channel_dict in image_sources.items():
        for channel_index, channel_name in enumerate(channel_names):
            logger.debug(
                f"Adding source for subpath {sub_path}, channel {channel_name}..."
            )
            plate_dataset.add_image_sources(
                path_dict=channel_dict[channel_name],
                channel_index=channel_index,
                data_format="ome.zarr",
            )
            merged_grid_name = f"merged_grid_{sub_path}_{channel_name}"
            plate_dataset.add_merged_grid(
                name=merged_grid_name,
                sources=list(channel_dict[channel_name]),
                positions=image_positions[sub_path][channel_name],
            )
            contrast_limits = channel_contrast_limits[sub_path][channel_name]
            if contrast_limits[0] == float("inf"):
                contrast_limits = [0.0, 255.0]
            plate_dataset.add_image_display(
                name=merged_grid_name,
                sources=[merged_grid_name],
                color=channel_colors[sub_path].get(channel_name, "white"),
                contrast_limits=(contrast_limits[0], contrast_limits[1]),
            )

    for sub_path, label_dict in label_sources.items():
        for label_name, source_dict in label_dict.items():
            logger.debug(f"Adding source for subpath {sub_path}, label {label_name}...")
            plate_dataset.add_segmentation_sources(
                path_dict=source_dict,
                table_path_dict=label_tables[sub_path][label_name],
                data_format="ome.zarr",
            )
            merged_grid_name = f"merged_grid_{sub_path}_{label_name}"
            plate_dataset.add_merged_grid(
                name=merged_grid_name,
                sources=list(source_dict.keys()),
                positions=label_positions[sub_path][label_name],
            )
            plate_dataset.add_segmentation_display(
                name=merged_grid_name,
                sources=[merged_grid_name],
                visible=False,
                color_by_column="label",
                additional_tables=additional_feature_tables[sub_path][label_name] or None,
            )

    # Add region display containing all wells
    plate_dataset.add_region_display(
        name="all_wells",
        map_of_sources={
            key: image_sources_per_well[key] + label_sources_per_well[key]
            for key in image_sources_per_well
        },
    )

    plate_dataset.save()
    project.save()
