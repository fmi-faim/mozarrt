"""Tests for create_plate_project."""

from pathlib import Path

import pandas as pd
import pytest
from mobiedantic import Dataset, Project
from ngio import create_empty_ome_zarr, create_empty_plate

from mozarrt.collect import create_plate_project

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

LABEL_NAME = "test_label"
WELL_SIZE = 20  # pixels in x and y
PIXEL_SIZE = 1.0  # µm per pixel


def _add_label(container, label_name: str = LABEL_NAME) -> None:
    """Add a label to an ngio container with two non-overlapping objects."""
    label = container.derive_label(label_name, overwrite=True)
    arr = label.get_as_numpy()
    arr[...] = 0
    arr[2:7, 2:7] = 1  # object 1 – top-left region
    arr[12:17, 12:17] = 2  # object 2 – bottom-right region
    label.set_array(arr)


@pytest.fixture(scope="module")
def minimal_plate(tmp_path_factory) -> Path:
    """Create a minimal 2-well plate: rows=A, cols=01/02, 1 channel, 1 label.

    Module-scoped to avoid repeated creation (mitigates Windows zarr PermissionErrors).
    """
    tmp_path = tmp_path_factory.mktemp("plate_data")
    plate_path = tmp_path / "test_plate.zarr"
    plate = create_empty_plate(plate_path, name="test_plate")
    for col in ("01", "02"):
        well = plate.add_well(row="A", column=col)
        image_store = well.add_image(image_path="0")
        container = create_empty_ome_zarr(
            image_store,
            shape=(1, WELL_SIZE, WELL_SIZE),
            axes_names=["c", "y", "x"],
            channels_meta=["DAPI"],
            levels=1,
            pixelsize=PIXEL_SIZE,
        )
        _add_label(container)
    return plate_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_create_plate_project_returns_project(minimal_plate, tmp_path):
    """create_plate_project should return a Project and write project.json."""
    out = tmp_path / "mobie"
    result = create_plate_project(minimal_plate, out)
    assert isinstance(result, Project)
    assert (out / "project.json").exists()


def test_dataset_folder_named_after_plate(minimal_plate, tmp_path):
    """Dataset folder should be named after the plate zarr directory."""
    out = tmp_path / "mobie"
    create_plate_project(minimal_plate, out)
    dataset_dir = out / minimal_plate.name
    assert dataset_dir.exists()
    assert (dataset_dir / "dataset.json").exists()


def test_intensity_and_segmentation_sources_created(minimal_plate, tmp_path):
    """Dataset should contain intensity sources and per-well segmentation sources."""
    out = tmp_path / "mobie"
    create_plate_project(minimal_plate, out)

    dataset = Dataset(out / minimal_plate.name)
    dataset.load()
    sources = dataset.model.sources

    # 2 intensity sources (one per well × one channel)
    intensity_sources = [n for n in sources if "DAPI" in n]
    assert len(intensity_sources) == 2

    # 2 segmentation sources (one per well × one label)
    seg_sources = [n for n in sources if LABEL_NAME in n]
    assert len(seg_sources) == 2


def test_combined_label_table_exists(minimal_plate, tmp_path):
    """A combined TSV table for the label name should exist in tables/."""
    out = tmp_path / "mobie"
    create_plate_project(minimal_plate, out)

    table_path = out / minimal_plate.name / "tables" / LABEL_NAME / "default.tsv"
    assert table_path.exists(), f"Table not found at {table_path}"


def test_combined_table_columns(minimal_plate, tmp_path):
    """Combined table must have label_id, anchor_x/y, bb columns, label_image_id."""
    out = tmp_path / "mobie"
    create_plate_project(minimal_plate, out)

    table_path = out / minimal_plate.name / "tables" / LABEL_NAME / "default.tsv"
    df = pd.read_csv(table_path, sep="\t")

    required = {
        "label_id",
        "anchor_x",
        "anchor_y",
        "label_image_id",
        "bb_min_x",
        "bb_min_y",
        "bb_max_x",
        "bb_max_y",
        "well",
        "plate_name",
    }
    assert required.issubset(df.columns), (
        f"Missing columns: {required - set(df.columns)}"
    )
    # well values should match well paths (e.g. "A/01", "A/02")
    assert set(df["well"].unique()) == {"A/01", "A/02"}
    # plate_name should match the plate zarr directory name
    assert (df["plate_name"] == minimal_plate.name).all()


def test_combined_table_row_count(minimal_plate, tmp_path):
    """Table should have 2 objects × 2 wells = 4 rows."""
    out = tmp_path / "mobie"
    create_plate_project(minimal_plate, out)

    table_path = out / minimal_plate.name / "tables" / LABEL_NAME / "default.tsv"
    df = pd.read_csv(table_path, sep="\t")
    assert len(df) == 4, f"Expected 4 rows, got {len(df)}"


def test_grid_offset_applied_correctly(minimal_plate, tmp_path):
    """
    Well A/02 (col_idx=1) has objects offset by (WELL_SIZE-1)*PIXEL_SIZE in x
    relative to A/01 (col_idx=0).  BDV uses (N-1)*scale as slot width.
    """
    out = tmp_path / "mobie"
    create_plate_project(minimal_plate, out)

    table_path = out / minimal_plate.name / "tables" / LABEL_NAME / "default.tsv"
    df = pd.read_csv(table_path, sep="\t")

    well01 = df[df["label_image_id"] == f"A01_{LABEL_NAME}"].sort_values("anchor_x")
    well02 = df[df["label_image_id"] == f"A02_{LABEL_NAME}"].sort_values("anchor_x")

    assert len(well01) == 2, f"Expected 2 rows for A/01, got {len(well01)}"
    assert len(well02) == 2, f"Expected 2 rows for A/02, got {len(well02)}"

    # col_idx for A/01 = 0, A/02 = 1 → offset_x = 1 * (WELL_SIZE-1) * PIXEL_SIZE
    # BDV uses (N-1)*scale as slot width (pixel-center to pixel-center extent)
    expected_offset = (WELL_SIZE - 1) * PIXEL_SIZE  # = 19.0
    for r1, r2 in zip(well01.itertuples(), well02.itertuples()):
        delta = r2.anchor_x - r1.anchor_x
        assert abs(delta - expected_offset) < 1e-6, (
            f"Expected anchor_x offset of {expected_offset}, got {delta}"
        )


def test_label_image_id_matches_per_well_source_name(minimal_plate, tmp_path):
    """label_image_id in each table row must match the per-well segmentation source name."""
    out = tmp_path / "mobie"
    create_plate_project(minimal_plate, out)

    table_path = out / minimal_plate.name / "tables" / LABEL_NAME / "default.tsv"
    df = pd.read_csv(table_path, sep="\t")

    # Per-well source names have the form "<row><col>_<label_name>"
    expected_ids = {f"A01_{LABEL_NAME}", f"A02_{LABEL_NAME}"}
    found_ids = set(df["label_image_id"].unique())
    assert found_ids == expected_ids, (
        f"Expected label_image_id values {expected_ids}, found {found_ids}"
    )


def test_segmentation_display_in_default_view(minimal_plate, tmp_path):
    """Default view should contain a SegmentationDisplay for the label."""
    out = tmp_path / "mobie"
    create_plate_project(minimal_plate, out)

    dataset = Dataset(out / minimal_plate.name)
    dataset.load()

    displays = dataset.model.views["default"].sourceDisplays
    assert displays is not None, "sourceDisplays should not be None"

    display_names = []
    for d in displays:
        d_dict = d.model_dump(exclude_none=True)
        if "segmentationDisplay" in d_dict:
            display_names.append(d_dict["segmentationDisplay"].get("name", ""))

    assert LABEL_NAME in display_names, (
        f"Expected SegmentationDisplay named '{LABEL_NAME}', found: {display_names}"
    )


def test_segmentation_source_tabledata_points_to_combined_table(
    minimal_plate, tmp_path
):
    """Each per-well segmentation source should point to the shared combined table dir."""
    out = tmp_path / "mobie"
    create_plate_project(minimal_plate, out)

    dataset = Dataset(out / minimal_plate.name)
    dataset.load()

    for source_name, source in dataset.model.sources.items():
        if LABEL_NAME not in source_name:
            continue
        src_dict = source.model_dump(exclude_none=True)
        assert "segmentation" in src_dict
        table_rel = src_dict["segmentation"]["tableData"]["tsv"]["relativePath"]
        # All label sources must point to the same shared table directory
        assert table_rel.endswith(f"tables/{LABEL_NAME}"), (
            f"Source {source_name} points to unexpected table dir: {table_rel}"
        )
