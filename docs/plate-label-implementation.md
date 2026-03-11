# Plate Label Annotation – Implementation Notes

## Overview

This document describes the implementation of HCS plate label annotation support in **mozarrt**. The goal is to annotate OME-Zarr labels from an HCS plate inside MoBIE/Fiji using a single combined annotation table per label name.

---

## Architecture

### Key design decisions

| Decision | Choice | Rationale |
|---|---|---|
| Table layout | One combined TSV per label name | Lets users annotate all labels across the whole plate in a single MoBIE table view |
| Anchor coordinates | Centroid + grid offset | Double-clicking a row in the table navigates to the correct segment in the correct well |
| `label_image_id` column | Present in every row | Tells MoBIE which per-well source each row belongs to |
| Segmentation display | `SegmentationDisplay` with `lut="glasbey"` | Required by MoBIE to render label overlays |
| Grid offset formula | `offset_x = col_idx × phys_width`, `offset_y = row_idx × phys_height` | Physical-unit offsets derived from pixel size and label shape |

### File layout (after running `create_plate_project`)

```
<output_directory>/
  project.json
  <plate_name>/
    dataset.json
    tables/
      <label_name>/          # one directory per unique label name
        default.tsv          # combined table: all wells, all objects
    <per-well intensity sources are registered with relative paths>
```

---

## New / changed files

### `src/mozarrt/_table_utils.py` (new)

Shared helpers used by both folder mode and plate mode.

**`source_path_payload(source_path, dataset_path, channel_index=None)`**  
Returns the `imageData` dict payload for a MoBIE source, using a relative path when possible and an absolute path as fallback.

**`normalize_relative_paths(value)`**  
Recursively walks a dict/list structure and converts backslashes to forward slashes in every `relativePath` value. Required on Windows.

**`compute_label_rows(label, *, label_image_id=None, offset_x=0, offset_y=0, offset_z=0)`**  
Reads the label array via ngio (`label.get_as_numpy()`), computes per-object centroids with `scipy.ndimage.center_of_mass` and bounding boxes with `find_objects`, then returns a list of row dicts.  

- Handles 2-D `(y, x)` and 3-D `(z, y, x)` arrays.  
- All spatial values are in physical units (centroid × pixel size).  
- When `label_image_id` is given, a `label_image_id` column is added to every row – this is how the combined plate table identifies which well each object came from.  
- `offset_x/y/z` are added to every coordinate so that rows for different wells end up at globally correct positions in the plate grid.

**`write_segmentation_table(rows, table_dir)`**  
Writes the list of row dicts to `table_dir/default.tsv` (tab-separated, no index). Creates the directory if needed.

**`add_segmentation_source(dataset, source_name, source_path, table_dir)`**  
Registers a segmentation source in a `mobiedantic.Dataset` model, pointing both `imageData` and `tableData` at the correct paths.

---

### `src/mozarrt/folder.py` (refactored)

The previously duplicated helpers (`_source_path_payload`, `_normalize_relative_paths`, `_write_segmentation_table`, `_add_segmentation_source`) were removed and replaced with imports from `_table_utils`. Functional behaviour is unchanged.

---

### `src/mozarrt/collect.py` – `create_plate_project` (extended)

The plate label pipeline is added after the existing intensity-channel loop.

#### Per-well label collection

For every well path in `plate.wells_paths()`:

1. Compute the grid position:

   ```python
   row_idx = plate.rows.index(row_letter)
   col_idx = plate.columns.index(col_number)
   ```

2. Open the field container and iterate over its labels.
3. Compute the physical well size:

   ```python
   phys_w = label.shape[-1] * label.pixel_size.x
   phys_h = label.shape[-2] * label.pixel_size.y
   ```

4. Compute the grid offset for this well:

   ```python
   offset_x = col_idx * phys_w
   offset_y = row_idx * phys_h
   ```

5. Call `compute_label_rows(label, label_image_id=seg_source_name, offset_x=…, offset_y=…)` and accumulate the rows in `label_rows[label_name]`.
6. Store `(seg_source_name, label_path)` in `label_source_names[label_name]`.

Label reading is wrapped in a `try/except` so that wells without labels are skipped with a warning rather than crashing.

#### Per-label-name finalisation

For each unique `label_name`:

1. Write the combined table:

   ```
   <dataset>/tables/<label_name>/default.tsv
   ```

2. Register every per-well segmentation source with `add_segmentation_source`, all pointing to the same shared `table_dir`.
3. Append a `SegmentationDisplay` to the default view:

   ```python
   SegmentationDisplay(
       segmentationDisplay=SegmentationDisplay1(
           name=label_name,
           sources=all_seg_source_names,
           opacity=0.5,
           lut="glasbey",
       )
   )
   ```

4. Apply `normalize_relative_paths` to all sources before saving.

---

### `src/mozarrt/cli.py` (updated)

`create_plate_project` is now registered as a CLI command under the `plate` sub-app:

```python
plate_app.command(create_plate_project)
```

Usage:

```
mozarrt plate create-plate-project <plate_zarr_path> <output_directory>
```

---

## Combined table format

`tables/<label_name>/default.tsv`

| Column | Type | Description |
|---|---|---|
| `label_id` | int | Label object ID within its well |
| `anchor_x` | float | Global centroid X in physical units |
| `anchor_y` | float | Global centroid Y in physical units |
| `anchor_z` | float | Global centroid Z (3-D only) |
| `bb_min_x` | float | Bounding-box minimum X |
| `bb_min_y` | float | Bounding-box minimum Y |
| `bb_max_x` | float | Bounding-box maximum X |
| `bb_max_y` | float | Bounding-box maximum Y |
| `label_image_id` | str | Source name of the per-well segmentation (e.g. `A01_test_label`) |

The `anchor_x/y` columns are required by `MoBIESegmentColumnNames.matches()` for MoBIE to recognise the file as a valid segmentation table. The `label_image_id` column is required for MoBIE to resolve multi-source segmentation displays.

---

## Tests – `tests/test_plate.py`

A `module`-scoped pytest fixture builds a minimal synthetic 2-well plate (`A/01`, `A/02`) using `ngio.create_empty_plate` and `ngio.create_empty_ome_zarr`. Each well field has one intensity channel (`DAPI`) and one label (`test_label`) with two objects placed at known pixel coordinates.

| Test | What it checks |
|---|---|
| `test_create_plate_project_returns_project` | Returns a `Project` object; `project.json` is written |
| `test_dataset_folder_named_after_plate` | Dataset folder name matches the plate zarr name |
| `test_intensity_and_segmentation_sources_created` | 2 intensity + 2 segmentation sources in `dataset.json` |
| `test_combined_label_table_exists` | `tables/test_label/default.tsv` is created |
| `test_combined_table_columns` | All required columns are present |
| `test_combined_table_row_count` | 2 objects × 2 wells = 4 rows |
| `test_grid_offset_applied_correctly` | `anchor_x` for well `A/02` is offset by exactly `WELL_SIZE × PIXEL_SIZE` relative to `A/01` |
| `test_segmentation_display_in_default_view` | A `SegmentationDisplay` named `test_label` exists in the default view |
| `test_segmentation_source_tabledata_points_to_combined_table` | Every per-well segmentation source's `tableData` points to `tables/test_label` |

All 15 tests (6 folder + 9 plate) pass.

---

## Dependencies

No new runtime dependencies were introduced. The implementation uses:

- **ngio** – plate/well/label access (`open_ome_zarr_plate`, `open_ome_zarr_container`)
- **mobiedantic** – MoBIE JSON schema (`SegmentationDisplay`, `SegmentationDisplay1`, `Source`, `Dataset`, `Project`)
- **scipy.ndimage** – `center_of_mass`, `find_objects` (already used in folder mode)
- **pandas** – TSV writing
