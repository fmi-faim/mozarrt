# Copilot instructions for `mozarrt`

## Build, test, and run

- Use Pixi for project commands and environments.
- Full test suite: `pixi run --environment py312 test`
- Other test environments used in CI: `pixi run --environment py313 test` and `pixi run --environment py314 test`
- Coverage XML: `pixi run --environment py312 cov-xml`
- Single test example: `pixi run --environment py312 pytest tests/test_folder.py::test_folder_dimension`
- Build a distribution: `pixi run --environment build build`

## High-level architecture

- `src/mozarrt/cli.py` wires a Cyclopts CLI with two sub-apps: `plate` and `folder`.
- There are two output paths: collection CSV tables listing images/label images, and MoBIE projects/datasets handled through `mobiedantic`.
- `src/mozarrt/collect.py` and `src/mozarrt/mobie_collection.py` build the collection-table rows used for CSV output.
- `src/mozarrt/plate.py`, `folder.py`, and `list.py` turn OME-Zarr inputs into MoBIE/MoBIEdantic `Project` and `Dataset` outputs.
- The package entrypoint is `mozarrt.cli:app`, exposed on the `mozarrt` console script.
- Tests use real OME-Zarr fixtures in `tests/resources/` plus synthetic plate data built with `ngio`.

## Key conventions

- Keep CLI entry functions in module namespaces and register them through `cyclopts.App` in `cli.py`.
- Use `cyclopts.types.ExistingDirectory` / `ExistingCsvPath` for filesystem arguments.
- Use `loguru.logger` for progress logging.
- Preserve the `Project`/`Dataset` flow: initialize the project, create the dataset, call `initialize_with_paths(...)`, add sources/views, then `save()` the dataset and project.
- Derive channel names from `image.channels_meta.channels`; do not hardcode channel labels or counts.
- Use `natsorted` when iterating over discovered `.zarr` directories so generated outputs stay deterministic.
- Tests assert generated MoBIE artifacts and inspect loaded `mobiedantic` models, not just raw files.
