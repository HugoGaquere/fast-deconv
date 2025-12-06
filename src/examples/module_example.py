import cupy as cp
import numpy as np
import fast_deconv

print(dir(fast_deconv))

size = 1000
data = cp.random.randn(size, size).astype(cp.float32)
mask = cp.random.rand(size, size) > 0.5  # ~50% ~50% False

resources = fast_deconv.stream_resources()

r = fast_deconv.matrix.argmax(data, mask, False, resources)
print(r)

breakpoint()
