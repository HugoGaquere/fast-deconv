import numpy as np

import fast_deconv

arr = np.arange(10, dtype=np.float32)
print("before", arr)
fast_deconv.add_one(arr)
print("after", arr)
