import numpy as np

import fast_deconv


def test_add_one():
    arr_orig = np.arange(10, dtype=np.float32)
    arr_copy = arr_orig.copy()
    fast_deconv.add_one(arr_copy)
    # arr_orig += 1
    np.testing.assert_equal(arr_copy, arr_orig + 1)
