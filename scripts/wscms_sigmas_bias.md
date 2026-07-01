# WSCMS — How DDFacet derives `sigmas` and `bias`

Reference: `DDFacet/Imager/WSCMS/ClassScaleMachine.py`.

Both arrays have shape `(Nscales,)`. They are derived from `alphas` (scale extent
in pixels) plus the `MultiScaleBias` config knob. `alphas` itself is either
user-supplied (`WSCMS.Scales`) or auto-derived from imaging parameters.

---

## 1. `alphas` — scale extents in pixels

Source: `ClassScaleMachine.set_scales()` (lines 366–417).

### User-supplied path

```python
alphas = np.asarray(GD["WSCMS"]["Scales"], dtype=float)
Nscales = alphas.size
```

### Auto-derivation path (`WSCMS.Scales is None`)

Inputs:
- `MaxBaseline` (max baseline length, in wavelengths at max frequency, passed to `Init()`)
- `Cell` from `GD["Image"]["Cell"]` — either a scalar or a `(cell_x, cell_y)` tuple, in arcsec
- `Npix_x, Npix_y` — image dimensions
- `MaxScale` from `GD["WSCMS"]["MaxScale"]` — defaults to `max(Npix_x, Npix_y) // 4`

```python
min_beam       = 1.0 / MaxBaseline                              # rad
cell_rad_x     = cell_x * pi / (180 * 3600)
cell_rad_y     = cell_y * pi / (180 * 3600)
FWHM0_pix_x    = sqrt(2) * min_beam / cell_rad_x                # fiddle factor sqrt(2)
FWHM0_pix_y    = sqrt(2) * min_beam / cell_rad_y                #   matches WSClean
alpha0         = ceil(mean(FWHM0_pix_x, FWHM0_pix_y) / 0.45)
if alpha0 % 2:        alpha0 += 1            # force alpha0 EVEN
alphas         = [0, alpha0, 4*alpha0]
while alphas[-1] < MaxScale:
    alphas.append(1.5 * alphas[-1])
alphas         = alphas[:-1]                  # drop the final element
```

Then a second parity pass over every non-zero alpha:

```python
for i in range(Nscales):
    if alphas[i] == 0: continue
    if alphas[i] % 2 == 0: alphas[i] += 1     # force later alphas ODD
```

Note: `alpha0` is forced even but every later alpha is forced odd. Looks
intentional (it survives both auto-derivation and user-supplied paths) so
preserve the asymmetry verbatim if porting.

---

## 2. `sigmas` — Gaussian std-dev per scale

Source: `ClassScaleMachine.set_kernels()` (lines 500–527). Trivially derived
from `alphas`:

```python
sigmas = zeros(Nscales, dtype=float64)
for i in range(Nscales):
    if alphas[i] == 0:
        sigmas[i] = 0.0              # delta scale, no kernel
    else:
        sigmas[i] = 3.0 * alphas[i] / 16.0
```

Dumped to `.npy` as `float32` via
`ClassImageDeconvMachineWSCMS._init_gpu_subminorloop`.

---

## 3. `bias` — Offringa multi-scale bias

Source: `ClassScaleMachine.set_bias()` (lines 487–498). Inputs:
- `alphas` (computed above)
- `MultiScaleBias` = `GD["WSCMS"]["MultiScaleBias"]` (called `beta` below)

Hardcoded constant: `first_auto_scale_size = beam_size_in_pixels * 2 = 8.0`.

```python
bias    = ones(Nscales, dtype=float64)
beta    = GD["WSCMS"]["MultiScaleBias"]
# bias[0] stays 1.0 (delta scale)
for i in range(1, Nscales):
    bias[i] = beta ** (-log2(alphas[i] / 8.0))
```

Dumped to `.npy` as `float32`.

---

## Summary — minimum inputs to derive both arrays in C++

If the caller already has `alphas` and `multi_scale_bias`:

```text
sigmas[i] = (alphas[i] == 0) ? 0 : 3 * alphas[i] / 16
bias[0]   = 1
bias[i]   = multi_scale_bias ** (-log2(alphas[i] / 8))   for i >= 1
```

Both are pure host arithmetic over `Nscales` (typically < 10).
