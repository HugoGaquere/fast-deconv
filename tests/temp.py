import nvtx
import cupy as cp
import fast_deconv as fd

a = cp.ones((8, 1, 20000, 20000), dtype=cp.float32)
b = cp.ones((8, 1, 20000, 20000), dtype=cp.float32)
c = cp.empty_like(a)
cp.cuda.runtime.deviceSynchronize()


resources = fd.stream_resources()
with nvtx.annotate("kernels"):
    fd.matrix.subtract(a[..., 500:1900, 600:1900], b[..., 500:1900, 600:1900], c[..., 500:1900, 600:1900], resources)
    cp.cuda.runtime.deviceSynchronize()
    cp.subtract(a[..., 500:1900, 600:1900], b[..., 500:1900, 600:1900], out=c[..., 500:1900, 600:1900])
    cp.cuda.runtime.deviceSynchronize()

slc = (slice(None), slice(None), slice(500, 1900), slice(600, 1900))
psf = a
dirty = b
coeffs = cp.ones(a.shape[0], dtype=cp.float32)
gain = 1.0
out = cp.empty_like(a)

cp.cuda.runtime.deviceSynchronize()

with nvtx.annotate("kernels"):
    fd.wscms.subtract_psf_from_dirty_async(psf[slc], dirty[slc], coeffs, out[slc], gain, resources)
    cp.cuda.runtime.deviceSynchronize()

    # Equivalent CuPy
    scaled_psf = psf[slc] * coeffs[:, None, None, None] * gain
    cp.subtract(dirty[slc], scaled_psf, out=out[slc])
    cp.cuda.runtime.deviceSynchronize()
