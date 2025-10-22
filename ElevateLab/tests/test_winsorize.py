import pandas as pd
import numpy as np
from First import remove_outliers


def test_winsorize_caps_values():
    df = pd.DataFrame({'A': [1, 2, 3, 1000], 'Survived': [0, 1, 0, 1]})
    df_w = remove_outliers(df, method='winsorize', k=1.5)
    # The maximum should be less than the original extreme 1000
    assert df_w['A'].max() < 1000
    # Length should be preserved
    assert len(df_w) == len(df)
