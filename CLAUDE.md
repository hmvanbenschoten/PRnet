# CLAUDE.md

## Input scale and normalization — do not remove

PRnet expects input expression on the **LINCS log2 scale** (`dataset/demo.h5ad`
ships with `X` spanning ~2.4 to 15.0; the CCLE path in
`preprocessing/custom_data_preprocessing.ipynb` uses log2(TPM+1)).

On top of that, every entry script applies:

```python
sc.pp.normalize_total(adata)
sc.pp.log1p(adata)
```

So already-log-scale data gets log-transformed again. This looks like a bug and
is not one: it is applied **identically at train and test time**, and the
shipped `checkpoint/*.pt` were trained that way (the `train_*.py` scripts have
carried both calls since 2024-03, five months before the checkpoints were
committed).

**Do not remove these lines without retraining from scratch.** Skipping them
drops per-gene-mean R² on the demo set from 0.88 to −8.27 and pushes decoder
output to 22.69 against a truth ceiling of 15.0.

`test_demo.py --skip_normalization` exists to demonstrate this, not as a
recommended mode.

See `KNOWN_ISSUES.md` §3 for the full history, measurements, and the one real
caveat (`normalize_total` is called without `target_sum`, so scaling depends on
the loaded subset's median library size).
