"""
Small demo script that:
- Loads `Titanic-Dataset.csv` if available
- Visualizes outliers (saves `outliers_boxplot.png`)
- Removes outliers using IQR
- Normalizes remaining numeric features
- Saves cleaned CSV to `Titanic-Dataset-Processed-wo-outliers.csv`

The script is defensive: it works if files or matplotlib are missing.
"""
from pathlib import Path
import os

import pandas as pd

from First import visualize_outliers, remove_outliers, normalize_features

DATAFILE = Path('Titanic-Dataset.csv')
OUT_PLOT = Path('outliers_boxplot.png')
OUT_CSV = Path('Titanic-Dataset-Processed-wo-outliers.csv')


def main():
    if not DATAFILE.exists():
        print(f"Data file not found at {DATAFILE}. Demo requires the Titanic CSV to run.")
        return

    df = pd.read_csv(DATAFILE)
    print(f"Loaded data with {len(df)} rows and {len(df.columns)} columns")

    # Visualize outliers and save plot
    fig = visualize_outliers(df, save_path=str(OUT_PLOT))
    if fig is not None:
        print(f"Saved outlier visualization to {OUT_PLOT}")
    else:
        print("No outlier visualization produced (matplotlib may be missing).")

    # Remove outliers
    df_clean = remove_outliers(df)
    print(f"Rows after outlier removal: {len(df_clean)}")

    # Normalize
    df_norm = normalize_features(df_clean)

    # Save
    df_norm.to_csv(OUT_CSV, index=False)
    print(f"Saved processed dataset to {OUT_CSV}")


if __name__ == '__main__':
    main()
