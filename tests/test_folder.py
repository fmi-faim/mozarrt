import json
import shutil
from pathlib import Path

import pandas as pd
import pytest
from mobiedantic import Dataset, Project
from ngio import open_ome_zarr_container

from mozarrt.folder import project


@pytest.mark.parametrize(
    "dimension,expected_is2d,num_sources",
    [
        ("2d", True, 13),
        ("3d", False, 9),
    ],
)
def test_folder_dimension(tmp_path, dimension, expected_is2d, num_sources):
    """Test that 2D/3D datasets are correctly identified and processed."""
    # Get the path to the test resources
    test_resources = Path(__file__).parent / "resources" / dimension

    # Run the project function
    project(test_resources, tmp_path, description=f"Test {dimension.upper()} project")

    # Verify that the project was created
    assert (tmp_path / "project.json").exists()

    test_project = Project(tmp_path)
    test_project.load()
    assert test_project.model.datasets[0].root == dimension

    # Verify that the dataset folder was created (named after input directory)
    dataset_dir = tmp_path / dimension
    assert dataset_dir.exists()
    assert (dataset_dir / "dataset.json").exists()

    # Load the dataset using mobiedantic to verify it's marked correctly
    dataset = Dataset(dataset_dir)
    dataset.load()
    assert dataset.model.is2D == expected_is2d

    # Verify that sources were created for all test datasets
    # We have 4 test datasets (test_data_0 to test_data_3), each with one channel
    assert len(dataset.model.sources) == num_sources


@pytest.mark.parametrize(
    "dimension,expected_table_rows",
    [
        ("2d", 4),
        ("3d", 4),
    ],
)
def test_folder_dimension_tables(tmp_path, dimension, expected_table_rows):
    """Test that the all_positions table is created for 2D/3D datasets."""
    test_resources = Path(__file__).parent / "resources" / dimension

    project(
        test_resources,
        tmp_path,
        description=f"Test {dimension.upper()} project with tables",
    )

    # Load the dataset using mobiedantic
    dataset_dir = tmp_path / dimension
    dataset = Dataset(dataset_dir)
    dataset.load()

    # Verify that the table was created
    table_path = dataset_dir / "tables" / "all_positions" / "default.tsv"
    assert table_path.exists()

    # Read the table and verify it has the correct number of positions
    table = pd.read_csv(table_path, sep="\t")
    assert len(table) == expected_table_rows


def test_folder_creates_segmentation_tables_for_observed_labels(tmp_path):
    """Test that observed label ids are written to segmentation default.tsv tables."""
    source_zarr = Path(__file__).parent / "resources" / "2d" / "test_data_0.zarr"
    input_dir = tmp_path / "label_input"
    input_dir.mkdir()
    target_zarr = input_dir / "sample.zarr"
    shutil.copytree(source_zarr, target_zarr)

    container = open_ome_zarr_container(target_zarr)
    label = container.derive_label("nuclei", overwrite=True)
    label_array = label.get_as_numpy()
    label_array[...] = 0
    label_array.flat[0] = 1
    label_array.flat[-1] = 5
    label.set_array(label_array)

    output_dir = tmp_path / "mobie_out"
    project(
        input_dir,
        output_dir,
        description="Test project with segmentation tables",
    )

    dataset_dir = output_dir / input_dir.name
    dataset = Dataset(dataset_dir)
    dataset.load()

    assert "sample_nuclei" in dataset.model.sources
    segmentation_source = dataset.model.sources["sample_nuclei"].model_dump(
        exclude_none=True
    )
    assert "segmentation" in segmentation_source

    table_rel = segmentation_source["segmentation"]["tableData"]["tsv"]["relativePath"]
    table_path = dataset_dir / table_rel / "default.tsv"
    assert table_path.exists()

    table = pd.read_csv(table_path, sep="\t")
    assert set(table["label_id"].tolist()) == {1, 5}

    with open(dataset_dir / "dataset.json") as dataset_file:
        dataset_json = json.load(dataset_file)

    region_sources = dataset_json["views"]["default"]["sourceDisplays"][0][
        "regionDisplay"
    ]["sources"]["sample"]
    assert "sample_nuclei" not in region_sources

    display_names = []
    for display in dataset_json["views"]["default"]["sourceDisplays"]:
        if "segmentationDisplay" in display:
            display_names.append(display["segmentationDisplay"]["name"])
    assert "sample_nuclei" in display_names
