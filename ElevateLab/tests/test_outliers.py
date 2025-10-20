import pandas as pd
import numpy as np
import os
import pytest

from First import remove_outliers, visualize_outliers


def make_df():
    # Create a dataframe with an obvious outlier in 'Fare'
    data = {
        'Age': [22, 30, 25, 40, 28],
        'Fare': [7.25, 8.05, 7.92, 9.5, 10000.0],  # last value is an outlier
        'SibSp': [1, 0, 0, 1, 0],
        'Parch': [0, 0, 0, 0, 0],
        'Survived': [0, 1, 1, 0, 1]
    }
    return pd.DataFrame(data)


def test_remove_outliers_iqr():
    df = make_df()
    df_clean = remove_outliers(df)
    # Ensure the outlier row is removed
    assert len(df_clean) < len(df)
    assert df_clean['Fare'].max() < 10000.0
    # Ensure Survived column stays present and intact for remaining rows
    assert 'Survived' in df_clean.columns


def test_visualize_outliers_returns_figure_or_none(tmp_path):
    df = make_df()
    # Try to create a plot file
    save_path = tmp_path / 'boxplot.png'
    fig = visualize_outliers(df, save_path=str(save_path))
    # If matplotlib is installed, fig should be a Figure; otherwise None
    assert (fig is None) or hasattr(fig, 'savefig')
    # If file was created, it should exist
    if save_path.exists():
        assert save_path.stat().st_size > 0
