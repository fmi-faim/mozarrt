# mozarrt

Tools for creating MoBIE-compatible project structures from folder-based OME-Zarr datasets.

## Current focus

This repo currently works best for **folder-based OME-Zarr** via the CLI command:

- `mozarrt folder project`

It builds a local MoBIE project directory (metadata, sources, grid views, and a position table) that you can open in Fiji + MoBIE.

If labels exist in an input OME-Zarr container, `mozarrt` now also creates:

- a MoBIE segmentation source for each label layer
- a segmentation table at `tables/<source_name>/default.tsv`
- one table row per observed non-zero label id

## Prerequisites

- [Pixi](https://pixi.sh/) installed
- Fiji installed
- MoBIE plugin available in Fiji (via Fiji updater / MoBIE update site)
- A directory containing `.zarr` datasets (e.g. one `.zarr` per field/position)

## Local install and test (development mode)

From the repository root:

1. Run tests to confirm the environment:

   - `pixi run -e py312 test`

2. Run the CLI on folder-based OME-Zarr test data:

   - `pixi run -e py312 mozarrt folder project tests/resources/2d ./tmp_mobie_project`

3. Verify output exists:

   - `./tmp_mobie_project/project.json`
   - `./tmp_mobie_project/2d/dataset.json`
   - `./tmp_mobie_project/2d/tables/all_positions/default.tsv`
   - (if labels exist) `./tmp_mobie_project/2d/tables/<zarr_stem>_<label_name>/default.tsv`

> Note: `pixi` uses editable package wiring from `pyproject.toml`, so local source changes in `src/mozarrt` are used directly.

## Test with your own folder-based OME-Zarr data

Replace paths with your own input/output:

- `pixi run -e py312 mozarrt folder project <input_folder_with_zarrs> <output_project_folder>`

Example:

- `pixi run -e py312 mozarrt folder project D:/data/my-plate-export D:/data/mobie-project`

## Open the generated project in Fiji (MoBIE)

1. Start Fiji.
2. Open MoBIE.
3. Choose to open a **local project**.
4. Select the generated project directory (the folder containing `project.json`).
5. Inspect sources/views and confirm `all_positions` table is present.

## High-level order of operations (`folder project`)

The command does the following:

1. Initialize a MoBIE project and dataset in the output directory.
2. Find all top-level `*.zarr` directories in the input folder.
3. Read the first OME-Zarr to infer:
   - 2D vs 3D (`is2D`)
   - channel labels
4. For each `.zarr` dataset:
   - create one source entry per channel
   - group sources by position
5. For each channel:
   - add all channel sources
   - add one merged grid view
6. Add a region/table view (`all_positions`) mapping each position to its sources.
7. For each detected label layer, create a segmentation source + `default.tsv` with observed label ids.
8. Save dataset and project metadata to disk.

## Troubleshooting

- If `pixi` is missing: install Pixi first and re-open terminal.
- If MoBIE fails to open the project: verify `project.json` exists at the selected path.
- If no data appears: confirm input folder actually contains top-level `*.zarr` directories.
- If CLI command is not found outside `pixi run`: run through `pixi run -e py312 ...` as shown above.
