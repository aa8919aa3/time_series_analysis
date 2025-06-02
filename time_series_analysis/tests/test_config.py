import pytest

def test_main_with_real_csv():
    from time_series_analysis.main import main
    result = main('path/to/real.csv')
    assert result is not None
    assert isinstance(result, dict)  # 假設 main() 返回一個字典
    assert 'expected_key' in result  # 根據實際情況檢查返回值的內容