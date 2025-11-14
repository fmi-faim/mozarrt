import pytest
from mobiedantic import Project, Dataset
from pathlib import Path

import pandas as pd

from mozarrt.folder import project


@pytest.mark.parametrize("dimension,expected_is2d,num_sources", [
    ("2d", True, 9),
    ("3d", False, 5),
])
def test_folder_dimension(tmp_path, dimension, expected_is2d, num_sources):
    """Test that 2D/3D datasets are correctly identified and processed."""
    # Get the path to the test resources
    test_resources = Path(__file__).parent / "resources" / dimension
    
    # Run the project function
    project(
        test_resources,
        tmp_path,
        description=f"Test {dimension.upper()} project"
    )
    
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


@pytest.mark.parametrize("dimension,expected_table_rows", [
    ("2d", 4),
    ("3d", 4),
])
def test_folder_dimension_tables(tmp_path, dimension, expected_table_rows):
    """Test that the all_positions table is created for 2D/3D datasets."""
    test_resources = Path(__file__).parent / "resources" / dimension
    
    project(
        test_resources,
        tmp_path,
        description=f"Test {dimension.upper()} project with tables"
    )
    
    # Load the dataset using mobiedantic
    dataset_dir = tmp_path / dimension
    dataset = Dataset(dataset_dir)
    dataset.load()
    
    # Verify that the table was created
    table_path = dataset_dir / "tables" / "all_positions" / "default.tsv"
    assert table_path.exists()
    
    # Read the table and verify it has the correct number of positions
    table = pd.read_csv(table_path, sep='\t')
    assert len(table) == expected_table_rows
