# Non-regression baselines

`ddmsc_synthetic.json` holds the expected scalar metrics of the synthetic
DDMSC non-regression run (`tests/cpp/test_ddmsc_nonreg.cu`). Values only —
per-metric tolerances live in the test code so a regeneration can never
loosen them silently.

Regenerate after an intentional algorithmic change:

```bash
FAST_DECONV_UPDATE_BASELINE=1 ctest --test-dir build/Release -L NONREG
git diff tests/baselines/ddmsc_synthetic.json   # review, then commit
```
