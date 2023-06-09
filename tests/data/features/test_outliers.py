import numpy as np
import pandas as pd

import kpe.data.features.outliers as cdto

def test_madn():
    df = pd.DataFrame([[1], [1], [2], [2], [4], [6], [9]], columns=['a'])
    x = cdto.mad(df)
    assert x == 1


def test_madn_2():
    df = pd.DataFrame([[1, 1], [1, 1], [2, 2], [2, 2], [4, 4], [6, 6], [9, 9]], columns=['a', 'b'])
    x = cdto.mad(df)
    assert np.equal(x, [1, 1]).all()


def test_mad_rule():
    arr = np.random.uniform(0, 1, size=(1000, 1))
    arr[0, 0] = 1000
    arr[1, 0] = -1000
    madm = cdto.MADMedianOutlierDetector()
    madm.fit(arr)
    vals = madm.rule_mask(arr)
    assert vals[0] and vals[1]
    assert vals.shape == (1000,)

def test_mad_rule():
    arr = np.random.uniform(0, 1, size=(1000, 1))
    arr[0, 0] = 1000
    arr[1, 0] = -1000
    madm = cdto.MADMedianOutlierDetector()
    vals = madm.fit_transform(arr)
    assert len(vals) == 998
