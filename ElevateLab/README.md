# Titanic Data Processing Demo

This workspace contains a small data processing pipeline for the Titanic dataset with helpers for:

- Missing-value handling
- Categorical encoding
- Normalization (sample std / ddof=1)
- Outlier handling (IQR removal or winsorization)
- Summary statistics (count, missing, mean, median, std, percentiles, skewness, kurtosis)
- Visualization: boxplots, histograms, KDE overlays, correlation heatmap, pairplot
- HTML report generation

Files of interest
- `First.py` - main library with processing and visualization functions
- `scripts/outlier_demo.py` - CLI demo to run pipeline and generate reports
- `tests/` - pytest tests covering key functions

Quick usage

1. Run the test suite:

```powershell
python -m pytest -q
```

2. Run the demo pipeline (uses `Titanic-Dataset.csv` in the repo):

```powershell
python .\scripts\outlier_demo.py --outlier IQR --save-stats summary_stats.csv --save-report report.html
```

3. Winsorize outliers and color pairplot by `Survived`:

```powershell
python .\scripts\outlier_demo.py --outlier winsorize --pairplot-hue Survived --save-report report.html
```

Notes
- Plotting features depend on `matplotlib`, `seaborn` and optionally `scipy` for KDE. If these aren't installed, plotting gracefully skips and the functions return `None`.
- `requirements.txt` includes plotting libs; install them with:

```powershell
pip install -r requirements.txt
```

If you'd like, I can:
- Add a GitHub Actions workflow to run tests and optionally build the HTML report
- Add more EDA plots (class-wise histograms, stacked bar charts)
- Add an interactive Jupyter notebook demo

Tell me which you prefer next.
