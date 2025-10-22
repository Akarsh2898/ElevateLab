"""Regenerate all EDA plots into plots/ directory.
Usage: python scripts/generate_plots.py
"""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent
DATA_CSV = ROOT / 'Titanic-Dataset.csv'
PLOTS_DIR = ROOT / 'plots'
PLOTS_DIR.mkdir(exist_ok=True)

print('Loading', DATA_CSV)
df = pd.read_csv(DATA_CSV)

# Basic numeric plots
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
for col in ['Fare']:
    fig, ax = plt.subplots(figsize=(8,4))
    sns.histplot(df[col].dropna(), bins=40, ax=ax)
    ax.set_title(f'{col} distribution')
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / f'{col.lower()}_hist.png', dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8,4))
    sns.histplot(np.log1p(df[col].dropna()), bins=40, ax=ax)
    ax.set_title(f'Log({col} + 1) distribution')
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / f'{col.lower()}_hist_log.png', dpi=150)
    plt.close(fig)

# Age by Survived
fig, ax = plt.subplots(figsize=(8,4))
sns.boxplot(x='Survived', y='Age', data=df, ax=ax)
sns.stripplot(x='Survived', y='Age', data=df, color='0.3', size=3, jitter=True, ax=ax)
ax.set_title('Age distribution by Survived')
ax.set_xticklabels(['No','Yes'])
fig.tight_layout()
fig.savefig(PLOTS_DIR / 'age_by_survived.png', dpi=150)
plt.close(fig)

# Correlation heatmap
numeric = df.select_dtypes(include=[np.number])
corr = numeric.corr()
fig, ax = plt.subplots(figsize=(6,5))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
ax.set_title('Correlation matrix (numeric)')
fig.tight_layout()
fig.savefig(PLOTS_DIR / 'corr_heatmap.png', dpi=150)
plt.close(fig)

# Categorical stacked plots
for col in ['Sex','Embarked']:
    fig, ax = plt.subplots(figsize=(6,4))
    stacked = df.groupby([col,'Survived']).size().unstack(fill_value=0)
    stacked.plot(kind='bar', stacked=True, ax=ax)
    ax.set_title(f'{col} counts stacked by Survived')
    ax.set_ylabel('Count')
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / f'{col}_by_Survived.png', dpi=150)
    plt.close(fig)

print('Plots regenerated in', PLOTS_DIR)
