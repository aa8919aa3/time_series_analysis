import pytest

def test_main_with_real_csv():
    assert main() is not None  # 假設 main() 函數返回一個非空值表示成功處理 CSV 檔案