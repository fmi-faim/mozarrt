from dataclasses import dataclass
from typing import Literal


@dataclass
class MoBIECollectionEntry:
    uri: str
    name: str | None = None
    type: Literal["intensities", "labels", "spots"] | None = None
    channel: int | None = None
    color: str | None = None
    blend: Literal["sum", "alpha"] | None = None
    affine: tuple[float] | None = None
    tps: str | None = None
    view: str | None = None
    exclusive: bool | None = None
    group: str | None = None
    contrast_limits: tuple[int, int] | None = None
    grid: str | None = None  # name of the grid
    grid_position: tuple[int, int] | None = None
    display: str | None = None
    format: Literal["OmeZarr"] | None = None
    spot_radius: float | None = None
    bounding_box: str | None = None


def collection_dataframe(entries: list[MoBIECollectionEntry]):
    """Convert a list of MoBIECollectionEntry to a pandas DataFrame."""
    import pandas as pd

    df = pd.DataFrame(entries)
    return df
