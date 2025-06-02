import pytest
import pandas as pd
from time_series_analysis.data_loader import load_data

def test_load_real_csv():
    df = load_data('path/to/real_data.csv')
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert 'expected_column' in df.columns