import pytest
from mobiedantic import Project, Dataset
from pathlib import Path
from ngio import open_ome_zarr_container

import pandas as pd

from mozarrt.folder import project


@pytest.mark.parametrize(
    "dimension,expected_is2d,num_sources",
    [
        ("2d", True, 9),
        ("3d", False, 5),
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


def _hex_rgb_to_rgba(color_hex: str) -> str:
    return (
        f"{int(color_hex[0:2], 16)}-"
        f"{int(color_hex[2:4], 16)}-"
        f"{int(color_hex[4:6], 16)}-255"
    )


@pytest.mark.parametrize("dimension", ["2d", "3d"])
def test_folder_image_display_uses_channel_metadata(tmp_path, dimension):
    test_resources = Path(__file__).parent / "resources" / dimension
    zarr_dirs = sorted(test_resources.glob("*.zarr"))
    first_container = open_ome_zarr_container(zarr_dirs[0])
    channel_names = [
        channel.label for channel in first_container.get_image().channels_meta.channels
    ]

    expected_color = {}
    expected_contrast_limits = {
        channel_name: [float("inf"), float("-inf")] for channel_name in channel_names
    }
    for zarr_dir in zarr_dirs:
        container = open_ome_zarr_container(zarr_dir)
        image = container.get_image()
        for channel_index, channel_name in enumerate(channel_names):
            channel_visualisation = container.meta.channels_meta.channels[
                channel_index
            ].channel_visualisation
            expected_color[channel_name] = _hex_rgb_to_rgba(channel_visualisation.color)
            image_channel_visualisation = image.meta.channels_meta.channels[
                channel_index
            ].channel_visualisation
            expected_contrast_limits[channel_name][0] = min(
                expected_contrast_limits[channel_name][0],
                image_channel_visualisation.start,
            )
            expected_contrast_limits[channel_name][1] = max(
                expected_contrast_limits[channel_name][1],
                image_channel_visualisation.end,
            )

    project(
        test_resources,
        tmp_path,
        description=f"Test {dimension.upper()} project with channel metadata",
    )

    dataset = Dataset(tmp_path / dimension)
    dataset.load()
    image_displays = {
        source_display.imageDisplay.name.root: source_display.imageDisplay
        for source_display in dataset.model.views["default"].sourceDisplays
        if hasattr(source_display, "imageDisplay")
    }

    for channel_name in channel_names:
        display = image_displays[f"merged_grid_{channel_name}"]
        assert display.color == expected_color[channel_name]
        assert display.contrastLimits == pytest.approx(
            expected_contrast_limits[channel_name]
        )
