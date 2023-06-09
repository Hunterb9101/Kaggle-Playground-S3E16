import pytest
import pandas as pd

import kpe.util as kpu

@pytest.mark.parametrize("name, expected", [
    ('str', str),
    ('kp.util.global_from_name', kpu.global_from_name),
    ('pandas.DataFrame', pd.DataFrame)
])
def test_global_from_name(name, expected):
    g = kpu.global_from_name(name)
    assert g == expected
