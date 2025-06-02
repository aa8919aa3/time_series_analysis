import pytest
import pandas as pd
from time_series_analysis.main import main

def test_main_with_real_csv():
    df = pd.read_csv('path/to/your/real.csv')
    result = main(df)
    expected_result = ...  # 根據實際情況填寫預期結果
    assert result == expected_result