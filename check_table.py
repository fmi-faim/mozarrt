import pandas as pd
from pathlib import Path

for lbl in ['sam_labels', 'ais_default_041225']:
    p = Path('tmp_plate/exp164-diff0.zarr/tables') / lbl / 'default.tsv'
    if p.exists():
        df = pd.read_csv(p, sep='\t')
        print(f'{lbl}: {len(df)} rows, unique label_image_ids: {df["label_image_id"].nunique()}')
        print('  first 5 label_image_id:', df['label_image_id'].head().tolist())
        print('  rows per well sample:')
        print(df['label_image_id'].value_counts().head(10))
        print()
    else:
        print(f'{lbl}: table not found at {p}')
