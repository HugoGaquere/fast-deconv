import cupy as cp
import fast_deconv

print(dir(fast_deconv))

size = 1000
A = cp.random.randn(size, size).astype(cp.float32)
B = cp.random.randn(size, size).astype(cp.float32)
C = cp.random.randn(size, size).astype(cp.float32)
mask = cp.random.rand(size, size) > 0.5  # ~50% ~50% False

resources = fast_deconv.stream_resources()

data_max = fast_deconv.matrix.argmax(A, mask, False, resources)
print(data_max)

fast_deconv.matrix.subtract(A, B, C, resources)

cp.testing.assert_array_equal(C, A-B)

breakpoint()
