import numpy as np
import pytest
from collections import defaultdict
from mobiedantic import Dataset, Project
from ngio import (
    create_empty_plate,
    create_ome_zarr_from_array,
    open_ome_zarr_plate,
    OmeZarrContainer,
    OmeZarrPlate,
)
from ngio.tables import FeatureTable
from pathlib import Path
import pandas as pd
import shutil

from mozarrt.plate import project

NGFF_VERSION = "0.4"


@pytest.fixture
def tmp_plate_zarr_path(tmp_path: Path) -> Path:
    """Fixture to create a temporary path for the OME-Zarr plate."""
    return tmp_path / "test_plate.zarr"


@pytest.fixture
def plate_dataset(tmp_plate_zarr_path: Path) -> OmeZarrPlate:
    """Fixture to create a synthetic OME-Zarr HCS plate dataset for testing."""
    rows = ["C", "D"]
    columns = ["3", "4"]
    common_shape = (2, 64, 64)  # (C, Y, X)
    levels = ["s0"]
    plate = create_empty_plate(
        store=tmp_plate_zarr_path,
        name="Test Plate",
        ngff_version=NGFF_VERSION,
        overwrite=True,
    )

    # add images to plate and keep well paths
    well_paths = []
    for row in rows:
        for column in columns:
            well_paths.append(
                plate.add_image(
                    row=row,
                    column=column,
                    image_path="fov0",
                )
            )

    well_containers: list[OmeZarrContainer] = []
    # random uint8 data for each well
    # set seed
    np.random.seed(42)
    for well_path in well_paths:
        image_array = np.random.randint(0, 256, size=common_shape, dtype=np.uint8)
        well_containers.append(
            create_ome_zarr_from_array(
                store=tmp_plate_zarr_path / well_path,
                array=image_array,
                axes_names="cyx",
                channels_meta=[
                    "RNA",
                    "DNA",
                ],
                levels=levels,
                pixelsize=0.5,
                ngff_version=NGFF_VERSION,
                overwrite=True,
            )
        )

    # create distinct label images for each well
    label_image_C03 = well_containers[0].derive_label(
        "object", ngff_version=NGFF_VERSION, channels_policy="singleton"
    )
    label_image_C03.set_array(
        _get_label_array((64, 64), [(32, 48, 8, 24), (8, 16, 48, 56)])[np.newaxis, ...],
    )
    label_image_C04 = well_containers[1].derive_label(
        "object", ngff_version=NGFF_VERSION, channels_policy="singleton"
    )
    label_image_C04.set_array(
        _get_label_array((64, 64), [])[np.newaxis, ...],  # empty label image
    )
    label_image_D03 = well_containers[2].derive_label(
        "object", ngff_version=NGFF_VERSION, channels_policy="singleton"
    )
    label_image_D03.set_array(
        _get_label_array((64, 64), [(0, 64, 0, 64)])[np.newaxis, ...],
    )
    label_image_D04 = well_containers[3].derive_label(
        "object", ngff_version=NGFF_VERSION, channels_policy="singleton"
    )
    label_image_D04.set_array(
        _get_label_array((64, 64), [(16, 48, 16, 48)])[np.newaxis, ...],
    )
    nuclei_image_C03 = well_containers[0].derive_label(
        "nuclei", ngff_version=NGFF_VERSION, channels_policy="singleton"
    )
    nuclei_image_C03.set_array(
        _get_label_array((64, 64), [(20, 28, 20, 28)])[np.newaxis, ...],
    )
    nuclei_image_C04 = well_containers[1].derive_label(
        "nuclei", ngff_version=NGFF_VERSION, channels_policy="singleton"
    )
    nuclei_image_C04.set_array(
        _get_label_array((64, 64), [(10, 18, 10, 18)])[np.newaxis, ...],
    )
    nuclei_image_D03 = well_containers[2].derive_label(
        "nuclei", ngff_version=NGFF_VERSION, channels_policy="singleton"
    )
    nuclei_image_D03.set_array(
        _get_label_array((64, 64), [])[np.newaxis, ...],
    )
    nuclei_image_D04 = well_containers[3].derive_label(
        "nuclei", ngff_version=NGFF_VERSION, channels_policy="singleton"
    )
    nuclei_image_D04.set_array(
        _get_label_array((64, 64), [(24, 40, 24, 40)])[np.newaxis, ...],
    )

    return plate


@pytest.fixture
def plate_dataset_with_feature_tables(
    tmp_plate_zarr_path: Path, plate_dataset: OmeZarrPlate
) -> OmeZarrPlate:
    """Fixture extending plate_dataset with FeatureTables on selected wells."""
    # Add a feature table referencing "object" label to C03 and D04
    image_dict = plate_dataset.get_images()
    # C03: 2 objects → feature table with 2 rows
    c03_feature_df = pd.DataFrame(
        {"label": [1, 2], "area": [128, 64], "intensity_mean": [150.0, 200.0]}
    )
    image_dict["C/03/fov0"].add_table(
        "morphology",
        FeatureTable(c03_feature_df, reference_label="object"),
    )
    # D04: 1 object → feature table with 1 row
    d04_feature_df = pd.DataFrame(
        {"label": [1], "area": [512], "intensity_mean": [175.0]}
    )
    image_dict["D/04/fov0"].add_table(
        "morphology",
        FeatureTable(d04_feature_df, reference_label="object"),
    )
    # C03: a nuclei feature table (for the nuclei label)
    c03_nuclei_df = pd.DataFrame({"label": [1], "area": [64]})
    image_dict["C/03/fov0"].add_table(
        "nuclei_morphology",
        FeatureTable(c03_nuclei_df, reference_label="nuclei"),
    )
    return plate_dataset


def _get_label_array(
    shape: tuple[int, int],
    labeled_regions: list[tuple[int, int, int, int]],
) -> np.ndarray:
    """Generate a label image array with distinct labels."""
    label_array = np.zeros(shape, dtype=np.uint16)
    for i, (x_start, x_end, y_start, y_end) in enumerate(labeled_regions, start=1):
        label_array[y_start:y_end, x_start:x_end] = i
    return label_array


def test_ngio_open_plate_dataset_fixture(
    plate_dataset: OmeZarrPlate, tmp_plate_zarr_path: Path
):
    """Test that the plate dataset fixture can be opened and contains expected data."""
    # Open the plate dataset
    opened_plate = open_ome_zarr_plate(tmp_plate_zarr_path)
    assert isinstance(opened_plate, OmeZarrPlate), (
        "Opened object is not an OmeZarrPlate."
    )


def test_plate_dataset_fixture(plate_dataset: OmeZarrPlate):
    """Base test for the table consolidation task."""
    image_dict = plate_dataset.get_images()
    assert len(image_dict) == 4, "Expected 4 images in the plate dataset."

    # assert that all dict keys are present
    assert list(image_dict.keys()) == [
        "C/03/fov0",
        "C/04/fov0",
        "D/03/fov0",
        "D/04/fov0",
    ]

    # assert that C/03 has one label image containing two distinct label values.
    # Labels are stored with a singleton leading dimension (1, Y, X) as per OME-Zarr convention.
    c_03_image = image_dict["C/03/fov0"]
    assert c_03_image.get_label("object").shape == (1, 64, 64)
    # distinct label values should be 0, 1, and 2 (background and two objects)
    assert set(np.unique(c_03_image.get_label("object").get_array())) == {0, 1, 2}

    c_04_image = image_dict["C/04/fov0"]
    assert c_04_image.get_label("object").shape == (1, 64, 64)
    # distinct label values should be 0 (background only)
    assert set(np.unique(c_04_image.get_label("object").get_array())) == {0}

    d_03_image = image_dict["D/03/fov0"]
    assert d_03_image.get_label("object").shape == (1, 64, 64)
    # distinct label values should be 1 (full image object)
    assert set(np.unique(d_03_image.get_label("object").get_array())) == {1}

    d_04_image = image_dict["D/04/fov0"]
    assert d_04_image.get_label("object").shape == (1, 64, 64)
    # distinct label values should be 0 and 1 (background and one object)
    assert set(np.unique(d_04_image.get_label("object").get_array())) == {0, 1}


def test_plate_project_creation(plate_dataset: OmeZarrPlate, tmp_path: Path):
    """Test that the plate project creation function runs without errors."""
    output_directory = tmp_path / "plate_project_output"
    output_directory.mkdir(exist_ok=True)
    project(
        tmp_path / "test_plate.zarr",
        output_directory,
    )
    # Check that the output JSON file was created
    assert (output_directory / "project.json").exists()


def test_plate_project_appends_dataset_to_existing_project(
    plate_dataset: OmeZarrPlate, tmp_path: Path
):
    output_directory = tmp_path / "plate_project_output_append"
    output_directory.mkdir(exist_ok=True)
    second_plate_path = tmp_path / "second_plate.zarr"
    shutil.copytree(tmp_path / "test_plate.zarr", second_plate_path)

    project(tmp_path / "test_plate.zarr", output_directory)
    project(second_plate_path, output_directory)

    mobie_project = Project(output_directory)
    mobie_project.load()
    dataset_names = [dataset.root for dataset in mobie_project.model.datasets]
    assert dataset_names == ["test_plate.zarr", "second_plate.zarr"]
    assert (output_directory / "test_plate.zarr" / "dataset.json").exists()
    assert (output_directory / "second_plate.zarr" / "dataset.json").exists()


def test_plate_project_fails_on_existing_dataset_without_force(
    plate_dataset: OmeZarrPlate, tmp_path: Path
):
    output_directory = tmp_path / "plate_project_output_collision"
    output_directory.mkdir(exist_ok=True)

    project(tmp_path / "test_plate.zarr", output_directory)

    with pytest.raises(ValueError, match="already exists in project"):
        project(tmp_path / "test_plate.zarr", output_directory)


def test_plate_project_force_overwrite_dataset(
    plate_dataset: OmeZarrPlate, tmp_path: Path
):
    output_directory = tmp_path / "plate_project_output_force_overwrite"
    output_directory.mkdir(exist_ok=True)

    project(tmp_path / "test_plate.zarr", output_directory)
    project(
        tmp_path / "test_plate.zarr",
        output_directory,
        force_overwrite_dataset=True,
    )

    mobie_project = Project(output_directory)
    mobie_project.load()
    dataset_names = [dataset.root for dataset in mobie_project.model.datasets]
    assert dataset_names == ["test_plate.zarr"]


def test_plate_segmentation_tables(plate_dataset: OmeZarrPlate, tmp_path: Path):
    """Test that per-well segmentation tables are generated with required columns.

    # TODO: Expand this test to cover all combinations of:
    #   - 2D label arrays (shape (Y, X))
    #   - 3D label arrays (shape (Z, Y, X) or singleton (1, Y, X))
    #   - with bbox measurements
    #   - without bbox measurements
    """
    output_directory = tmp_path / "plate_project_output"
    output_directory.mkdir(exist_ok=True)
    project(
        tmp_path / "test_plate.zarr",
        output_directory,
    )

    dataset_dir = output_directory / "test_plate.zarr"
    expected_rows_per_table = {
        "C03_object": 2,
        "C04_object": 0,
        "D03_object": 1,
        "D04_object": 1,
    }
    # Labels are stored as (1, Y, X) — regionprops_table treats them as 3D,
    # producing 3-component centroids and 6 bbox columns.
    required_columns = [
        "label",
        "centroid-0",
        "centroid-1",
        "centroid-2",
        "bbox-0",
        "bbox-1",
        "bbox-2",
        "bbox-3",
        "bbox-4",
        "bbox-5",
    ]

    for table_name, expected_rows in expected_rows_per_table.items():
        table_path = dataset_dir / "tables" / table_name / "default.tsv"
        assert table_path.exists()
        table = pd.read_csv(table_path, sep="\t")
        assert list(table.columns) == required_columns
        assert len(table) == expected_rows

    dataset = Dataset(dataset_dir)
    dataset.load()
    dataset_model_json = dataset.model.model_dump_json(by_alias=True)

    for table_name in expected_rows_per_table:
        assert f"tables/{table_name}" in dataset_model_json
        assert f"tables/{table_name}/default.tsv" not in dataset_model_json


def test_plate_project_exclude_labels(plate_dataset: OmeZarrPlate, tmp_path: Path):
    output_directory = tmp_path / "plate_project_output_excluded"
    output_directory.mkdir(exist_ok=True)
    project(
        tmp_path / "test_plate.zarr",
        output_directory,
        exclude_labels=["nuclei"],
    )

    dataset_dir = output_directory / "test_plate.zarr"
    for table_name in ["C03_nuclei", "C04_nuclei", "D03_nuclei", "D04_nuclei"]:
        table_path = dataset_dir / "tables" / table_name / "default.tsv"
        assert not table_path.exists()

    for table_name in ["C03_object", "C04_object", "D03_object", "D04_object"]:
        table_path = dataset_dir / "tables" / table_name / "default.tsv"
        assert table_path.exists()

    dataset = Dataset(dataset_dir)
    dataset.load()
    dataset_model_json = dataset.model.model_dump_json(by_alias=True)
    assert "merged_grid_fov0_nuclei" not in dataset_model_json
    assert "merged_grid_fov0_object" in dataset_model_json


def test_plate_project_exclude_labels_ignores_missing(
    plate_dataset: OmeZarrPlate, tmp_path: Path
):
    output_directory = tmp_path / "plate_project_output_missing_excluded"
    output_directory.mkdir(exist_ok=True)
    project(
        tmp_path / "test_plate.zarr",
        output_directory,
        exclude_labels=["missing-label"],
    )

    dataset_dir = output_directory / "test_plate.zarr"
    for table_name in ["C03_object", "C04_object", "D03_object", "D04_object"]:
        table_path = dataset_dir / "tables" / table_name / "default.tsv"
        assert table_path.exists()


def _hex_rgb_to_rgba(color_hex: str) -> str:
    return (
        f"{int(color_hex[0:2], 16)}-"
        f"{int(color_hex[2:4], 16)}-"
        f"{int(color_hex[4:6], 16)}-255"
    )


def test_plate_image_display_uses_channel_metadata(
    plate_dataset: OmeZarrPlate, tmp_path: Path
):
    output_directory = tmp_path / "plate_project_output_channel_metadata"
    output_directory.mkdir(exist_ok=True)

    first_well = plate_dataset.get_well(
        row=plate_dataset.rows[0], column=plate_dataset.columns[0]
    )
    first_image = first_well.get_image(first_well.paths()[0]).get_image()
    channel_names = [channel.label for channel in first_image.channels_meta.channels]

    expected_color = defaultdict(dict)
    expected_contrast_limits = defaultdict(
        lambda: defaultdict(lambda: [float("inf"), float("-inf")])
    )
    for _, well_object in plate_dataset.get_wells().items():
        for sub_path in well_object.paths():
            image_container = well_object.get_image(sub_path)
            image = image_container.get_image()
            for channel_index, channel_name in enumerate(channel_names):
                channel_visualisation = image_container.meta.channels_meta.channels[
                    channel_index
                ].channel_visualisation
                expected_color[sub_path][channel_name] = _hex_rgb_to_rgba(
                    channel_visualisation.color
                )
                image_channel_visualisation = image.meta.channels_meta.channels[
                    channel_index
                ].channel_visualisation
                expected_contrast_limits[sub_path][channel_name][0] = min(
                    expected_contrast_limits[sub_path][channel_name][0],
                    image_channel_visualisation.start,
                )
                expected_contrast_limits[sub_path][channel_name][1] = max(
                    expected_contrast_limits[sub_path][channel_name][1],
                    image_channel_visualisation.end,
                )

    project(
        tmp_path / "test_plate.zarr",
        output_directory,
    )

    dataset = Dataset(output_directory / "test_plate.zarr")
    dataset.load()
    image_displays = {
        source_display.imageDisplay.name.root: source_display.imageDisplay
        for source_display in dataset.model.views["default"].sourceDisplays
        if hasattr(source_display, "imageDisplay")
    }

    for sub_path, sub_path_channel_meta in expected_contrast_limits.items():
        for channel_name, contrast_limits in sub_path_channel_meta.items():
            display_name = f"merged_grid_{sub_path}_{channel_name}"
            display = image_displays[display_name]
            assert display.color == expected_color[sub_path][channel_name]
            assert display.contrastLimits == pytest.approx(contrast_limits)


def test_plate_merged_grid_positions_follow_well_layout(
    plate_dataset: OmeZarrPlate, tmp_path: Path
):
    output_directory = tmp_path / "plate_project_output_merged_grid_positions"
    output_directory.mkdir(exist_ok=True)

    project(
        tmp_path / "test_plate.zarr",
        output_directory,
    )

    dataset = Dataset(output_directory / "test_plate.zarr")
    dataset.load()
    source_transforms = dataset.model.views["default"].sourceTransforms

    merged_grids = {
        transform.mergedGrid.mergedGridSourceName.root: transform.mergedGrid
        for transform in source_transforms
    }

    expected_positions = {
        "C03_fov0_RNA": (0, 0),
        "C04_fov0_RNA": (1, 0),
        "D03_fov0_RNA": (0, 1),
        "D04_fov0_RNA": (1, 1),
    }
    rna_merged_grid = merged_grids["merged_grid_fov0_RNA"]
    assert {
        source.root: tuple(position)
        for source, position in zip(
            rna_merged_grid.sources, rna_merged_grid.positions.root
        )
    } == expected_positions

    expected_positions = {
        "C03_fov0_object": (0, 0),
        "C04_fov0_object": (1, 0),
        "D03_fov0_object": (0, 1),
        "D04_fov0_object": (1, 1),
    }
    object_merged_grid = merged_grids["merged_grid_fov0_object"]
    assert {
        source.root: tuple(position)
        for source, position in zip(
            object_merged_grid.sources, object_merged_grid.positions.root
        )
    } == expected_positions


def test_plate_include_feature_tables(
    plate_dataset_with_feature_tables: OmeZarrPlate, tmp_path: Path
):
    """Test that FeatureTables are written as additional TSV files and registered
    as additionalTables in the MoBIE segmentation display."""
    output_directory = tmp_path / "plate_project_output_feature_tables"
    output_directory.mkdir(exist_ok=True)

    project(
        tmp_path / "test_plate.zarr",
        output_directory,
        include_feature_tables=True,
    )

    dataset_dir = output_directory / "test_plate.zarr"

    # C03 and D04 have a "morphology" feature table for "object" label
    # C03 also has a "nuclei_morphology" feature table for "nuclei" label
    assert (dataset_dir / "tables" / "C03_object" / "morphology.tsv").exists()
    morphology_c03 = pd.read_csv(
        dataset_dir / "tables" / "C03_object" / "morphology.tsv", sep="\t"
    )
    assert morphology_c03.columns[0] == "label"
    assert set(morphology_c03.columns) == {"label", "area", "intensity_mean"}
    assert len(morphology_c03) == 2

    assert (dataset_dir / "tables" / "D04_object" / "morphology.tsv").exists()
    morphology_d04 = pd.read_csv(
        dataset_dir / "tables" / "D04_object" / "morphology.tsv", sep="\t"
    )
    assert len(morphology_d04) == 1

    # nuclei label feature table should be in the nuclei table directory
    assert (dataset_dir / "tables" / "C03_nuclei" / "nuclei_morphology.tsv").exists()

    # Wells without a feature table should not have extra files
    assert not (dataset_dir / "tables" / "C04_object" / "morphology.tsv").exists()
    assert not (dataset_dir / "tables" / "D03_object" / "morphology.tsv").exists()

    # The MoBIE model should list the additional tables in the segmentation display
    dataset = Dataset(dataset_dir)
    dataset.load()
    dataset_model_json = dataset.model.model_dump_json(by_alias=True)
    assert "morphology.tsv" in dataset_model_json
    assert "nuclei_morphology.tsv" in dataset_model_json


def test_plate_include_feature_tables_disabled_by_default(
    plate_dataset_with_feature_tables: OmeZarrPlate, tmp_path: Path
):
    """Test that FeatureTables are NOT included when include_feature_tables=False (default)."""
    output_directory = tmp_path / "plate_project_output_no_feature_tables"
    output_directory.mkdir(exist_ok=True)

    project(
        tmp_path / "test_plate.zarr",
        output_directory,
    )

    dataset_dir = output_directory / "test_plate.zarr"

    # No additional table files should exist
    assert not (dataset_dir / "tables" / "C03_object" / "morphology.tsv").exists()
    assert not (dataset_dir / "tables" / "D04_object" / "morphology.tsv").exists()
    assert not (dataset_dir / "tables" / "C03_nuclei" / "nuclei_morphology.tsv").exists()

    # The MoBIE model should not list any additional tables
    dataset = Dataset(dataset_dir)
    dataset.load()
    dataset_model_json = dataset.model.model_dump_json(by_alias=True)
    assert "morphology.tsv" not in dataset_model_json
    assert "nuclei_morphology.tsv" not in dataset_model_json


def test_plate_include_feature_tables_respects_exclude_labels(
    plate_dataset_with_feature_tables: OmeZarrPlate, tmp_path: Path
):
    """Test that FeatureTables for excluded labels are not included."""
    output_directory = tmp_path / "plate_project_output_ft_exclude"
    output_directory.mkdir(exist_ok=True)

    project(
        tmp_path / "test_plate.zarr",
        output_directory,
        include_feature_tables=True,
        exclude_labels=["nuclei"],
    )

    dataset_dir = output_directory / "test_plate.zarr"

    # object tables and feature tables should still be there
    assert (dataset_dir / "tables" / "C03_object" / "morphology.tsv").exists()
    assert (dataset_dir / "tables" / "D04_object" / "morphology.tsv").exists()

    # nuclei feature table directory should not exist (label was excluded)
    assert not (dataset_dir / "tables" / "C03_nuclei").exists()

    dataset = Dataset(dataset_dir)
    dataset.load()
    dataset_model_json = dataset.model.model_dump_json(by_alias=True)
    assert "morphology.tsv" in dataset_model_json
    assert "nuclei_morphology.tsv" not in dataset_model_json

