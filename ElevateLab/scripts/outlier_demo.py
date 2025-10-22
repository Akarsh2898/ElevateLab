"""
CLI demo script for processing the Titanic dataset and generating reports.

Usage examples:
  # Run pipeline with IQR outlier removal, save stats and report
  python scripts/outlier_demo.py --outlier IQR --save-stats summary.csv --save-report report.html

  # Winsorize outliers and generate pairplot colored by Survived
  python scripts/outlier_demo.py --outlier winsorize --pairplot-hue Survived

Options are defensive: plotting works if matplotlib/seaborn are installed; otherwise it's skipped.
"""
import argparse
from pathlib import Path
import sys
import os

# Ensure the repository root is on sys.path so `from First import ...` works when running
# this script directly (sys.path[0] is the script directory in that case).
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd

from First import (
    analyze_and_handle_missing_values,
    remove_outliers,
    encode_categorical_features,
    normalize_features,
    generate_summary_statistics,
    generate_html_report,
)

DATAFILE = Path('Titanic-Dataset.csv')


def run_pipeline(datafile, outlier=None, outlier_k=1.5, save_stats=None, save_report=None, pairplot_hue=None, pairplot_max_vars=8):
    if not datafile.exists():
        print(f"Data file not found at {datafile}")
        return

    df = pd.read_csv(datafile)
    print(f"Loaded: {len(df)} rows, {len(df.columns)} columns")

    # Missing values
    df_clean = analyze_and_handle_missing_values(df)

    # Outliers
    if outlier:
        if outlier not in ('IQR', 'winsorize'):
            print("Unsupported outlier method; choose 'IQR' or 'winsorize'")
        else:
            df_clean = remove_outliers(df_clean, method=outlier, k=outlier_k)
            print(f"After outlier handling: {len(df_clean)} rows")

    # Encode and normalize
    df_encoded = encode_categorical_features(df_clean)
    df_norm = normalize_features(df_encoded)

    processed_path = Path('Titanic-Dataset-Processed-wo-outliers.csv')
    df_norm.to_csv(processed_path, index=False)
    print(f"Saved processed data to {processed_path}")

    # Summary stats
    if save_stats:
        try:
            generate_summary_statistics(df_clean, save_path=save_stats)
            print(f"Saved stats to {save_stats}")
        except Exception as e:
            print(f"Failed to save stats: {e}")

    # HTML report
    if save_report:
        try:
            generate_html_report(df_clean, save_path=save_report, pairplot_hue=pairplot_hue, pairplot_max_vars=pairplot_max_vars)
            print(f"Saved report to {save_report}")
        except Exception as e:
            print(f"Failed to save report: {e}")


def main(argv=None):
    parser = argparse.ArgumentParser(description='Demo pipeline for Titanic preprocessing and reports')
    parser.add_argument('--datafile', default=str(DATAFILE), help='CSV input file')
    parser.add_argument('--outlier', choices=['IQR', 'winsorize'], help='Outlier handling method')
    parser.add_argument('--outlier-k', type=float, default=1.5, help='IQR multiplier')
    parser.add_argument('--save-stats', help='Path to save summary statistics CSV')
    parser.add_argument('--save-report', help='Path to save HTML report')
    parser.add_argument('--pairplot-hue', help='Column to use as hue for pairplot')
    parser.add_argument('--pairplot-max-vars', type=int, default=8, help='Max variables for pairplot')

    args = parser.parse_args(argv)
    run_pipeline(
        Path(args.datafile),
        outlier=args.outlier,
        outlier_k=args.outlier_k,
        save_stats=args.save_stats,
        save_report=args.save_report,
        pairplot_hue=args.pairplot_hue,
        pairplot_max_vars=args.pairplot_max_vars,
    )


if __name__ == '__main__':
    main()
