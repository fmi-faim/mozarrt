"""Remove 'sam_labels' entries from .zattrs files in zarr label folders."""

import json
import shutil
from pathlib import Path

from tqdm import tqdm

ROOT = Path(r"./data/exp181-diff8.zarr")
print(ROOT)

for labels_dir in tqdm(list(ROOT.rglob("labels"))):
    zattrs = labels_dir / ".zattrs"
    if zattrs.exists():
        data = json.loads(zattrs.read_text())
        labels = data.get("labels", [])
        if "sam_labels" in labels:
            data["labels"] = [label for label in labels if label != "sam_labels"]
            zattrs.write_text(json.dumps(data, indent=2))
            print(f"Updated: {zattrs}")

    sam_labels_dir = labels_dir / "sam_labels"
    if sam_labels_dir.exists() and sam_labels_dir.is_dir():
        shutil.rmtree(sam_labels_dir)
        print(f"Deleted: {sam_labels_dir}")
