import pandas as pd
import numpy as np
from First import generate_summary_statistics


def make_df():
    return pd.DataFrame({
        'A': [1, 2, 3, np.nan, 5],
        'B': [10, 20, 30, 40, 50],
        'Survived': [0, 1, 0, 1, 0]
    })


def test_summary_basic(tmp_path):
    df = make_df()
    stats = generate_summary_statistics(df)
    # Should contain rows for A and B, not Survived
    assert 'A' in stats.index
    assert 'B' in stats.index
    assert 'Survived' not in stats.index
    # Check some columns
    assert stats.loc['A', 'count'] == 4
    assert round(stats.loc['B', 'mean'], 6) == 30.0


def test_summary_save_csv(tmp_path):
    df = make_df()
    out = tmp_path / 'stats.csv'
    stats = generate_summary_statistics(df, save_path=str(out))
    assert out.exists()
    loaded = pd.read_csv(out, index_col=0)
    assert 'mean' in loaded.columns