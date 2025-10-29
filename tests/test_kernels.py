import cupy as cp
import numpy as np
from fast_deconv import kernels

def test_wscms():
    wscms = kernels.WSCMS()
    import numpy as np
    import numpy.fft as fft

    X = np.random.rand(100, 100)
    x, y = 10, 10
    XF = fft.fft(X)
    print(x, y, )
    print(XF[x, y])

    breakpoint()

test_wscms()


def test_argmax():
    for _ in range(100):
        size = 10_000_000
        data = cp.random.randn(size).astype(cp.float32)
        mask = cp.random.rand(size) > 0.5  # ~50% True, ~50% False

        masked_data = cp.where(mask, data, -cp.inf)
        true_index = cp.argmax(masked_data)
        true_value = data[true_index]

        actual_index, actual_value = kernels.argmax(data, mask, False)

        assert true_index == actual_index
        assert true_value == actual_value


def test_argmax_abs():
    for _ in range(100):
        size = 10_000_000
        data = cp.random.randn(size).astype(cp.float32)
        mask = cp.random.rand(size) > 0.5  # ~50% ~50% False

        masked_data = cp.where(mask, abs(data), 0)
        true_index = cp.argmax(masked_data)
        true_value = masked_data[true_index]

        actual_index, actual_value = kernels.argmax(data, mask, True)

        assert true_index == actual_index
        assert true_value == actual_value

