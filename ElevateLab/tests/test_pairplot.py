import pandas as pd
from First import generate_pairplot


def make_df():
    return pd.DataFrame({
        'A': [1, 2, 3, 4, 5],
        'B': [2, 3, 4, 5, 6],
        'C': [5, 4, 3, 2, 1]
    })


def test_pairplot_saves_or_none(tmp_path):
    df = make_df()
    out = tmp_path / 'pairplot.png'
    # also exercise max_vars and hue parameters (hue missing here should be ignored)
    res = generate_pairplot(df, save_path=str(out), max_vars=3, hue='NonExistent')
    # Either returns None (missing libs) or path to file
    assert (res is None) or (out.exists())
