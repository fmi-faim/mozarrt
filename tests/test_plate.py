import numpy as np
import pytest
from mobiedantic import Dataset, Project
from ngio import (
    create_empty_plate,
    create_ome_zarr_from_array,
    open_ome_zarr_plate,
    OmeZarrContainer,
    OmeZarrPlate,
)
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
    c_03_image = image_dict["C/03/fov0"]
    assert c_03_image.get_label("object").shape == (64, 64)
    # distinct label values should be 0, 1, and 2 (background and two objects)
    assert set(np.unique(c_03_image.get_label("object").get_array())) == {0, 1, 2}

    c_04_image = image_dict["C/04/fov0"]
    assert c_04_image.get_label("object").shape == (64, 64)
    # distinct label values should be 0 (background only)
    assert set(np.unique(c_04_image.get_label("object").get_array())) == {0}

    d_03_image = image_dict["D/03/fov0"]
    assert d_03_image.get_label("object").shape == (64, 64)
    # distinct label values should be 1 (full image object)
    assert set(np.unique(d_03_image.get_label("object").get_array())) == {1}

    d_04_image = image_dict["D/04/fov0"]
    assert d_04_image.get_label("object").shape == (64, 64)
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
    """Test that per-well segmentation tables are generated with required columns."""
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
    required_columns = ["label", "centroid-0", "centroid-1"]

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
