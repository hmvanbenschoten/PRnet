# Known issues

Issues found while porting this fork to Python 3.11 / PyTorch 2.x that were
**deliberately left unfixed**, because fixing them would change model behaviour,
training dynamics, or published numbers. They are recorded here so the next
person does not have to rediscover them.

The compatibility port itself touched only device handling, removed API kwargs,
removed sparse-matrix aliases and dead imports; it did not alter the model,
the loss, or the training loop.

---

## 1. Gradient flow is broken through the MSE and KL loss terms

`trainer/PRnetTrainer.py`, training loop.

The `GUSS`/`NB` reconstruction loss is computed from `gene_means` / `gene_vars`
and is differentiable. But the optional `MSE` and `KL` terms are computed
against `nb_sample`, which comes from `dist.sample()`:

```python
nb_sample = dist.sample()
if set(['MSE']).issubset(self.loss):
    mse_loss = self.mse_loss(nb_sample, target)
    reconstruction_loss += mse_loss * 10
if set(['KL']).issubset(self.loss):
    kl_loss = self.kl_loss(nb_sample, target)
    reconstruction_loss += kl_loss * 0.01
```

`.sample()` does not implement the reparameterisation trick (that would be
`.rsample()`), so it returns a tensor detached from the graph. Those terms
therefore contribute a constant to the reported loss and **no gradient at all**.
Any run configured with `loss=['GUSS','MSE']` is effectively training on `GUSS`
alone, with an inflated loss number.

This does not affect the default configurations in this repo — every entry
script uses `'loss': ['GUSS']` — but it does mean the `MSE`/`KL` options are
non-functional rather than merely unused.

Two further notes on the same block:

- `self.kl_loss` is `nn.KLDivLoss(reduction='batchmean')`, which expects
  log-probabilities as input and probabilities as target. It is being handed
  raw expression values, so even with gradients restored the quantity would not
  be a KL divergence.
- `torch.Tensor(gene_means)` and `torch.Tensor(gene_vars.sqrt())` re-wrap
  tensors that are already on the target device. This is a redundant copy, and
  on a non-CPU device it is also a silent host round-trip.

**Fix would change:** loss magnitudes and the optimisation path for any
`MSE`/`KL` run. Needs re-tuning of the `* 10` and `* 0.01` weights.

---

## 2. `shuffle_adata()` does not shuffle

`trainer/_utils.py`.

```python
ind_list = [i for i in range(adata.shape[0])]
shuffle(ind_list)
new_adata = adata[ind_list, :]
```

Two problems compound:

1. `shuffle` here is `random.shuffle` from the stdlib (imported at the top of
   the file), not a numpy RNG, so it is not covered by the `torch.manual_seed`
   call in `PRnetTrainer.__init__` and the permutation is unseeded.
2. More importantly, `ind_list` holds *positions*, but the result is only ever
   consumed by label-based lookups. `train_valid_test()` immediately does
   `shuffled[train_index, :]` where `train_index` is a list of `.obs` index
   **labels** taken from the *unshuffled* `adata`. AnnData resolves those labels
   against the shuffled view, so the original rows come back regardless of the
   permutation.

The net effect is that the shuffle is a no-op. This is benign in practice —
both `DataLoader`s that matter are constructed with `shuffle=True` — but the
function is misleading and the unseeded RNG would become a reproducibility
problem the moment anyone relied on it.

**Fix would change:** row ordering, hence batch composition, hence the exact
numbers from any re-run.

---

## 3. Normalisation is applied unconditionally in the entry scripts

`train_lincs.py`, `train_sciplex.py`, `test_lincs.py`, `test_sciplex.py`,
`test_demo.py` all do:

```python
adata = sc.read(...)
sc.pp.normalize_total(adata)
sc.pp.log1p(adata)
```

This was added upstream in commit `f19174b` (2024-12-13, "add normalization and
log-tranformation"). There is no guard against input that is **already**
normalised — and the datasets distributed with the paper, plus the output of
`preprocessing/custom_data_preprocessing.ipynb`, are already log-scale. CCLE
expression in that notebook is log2(TPM+1); running `normalize_total` + `log1p`
over it log-transforms twice.

Consequences:

- Any dataset preprocessed before that commit is now being double-transformed.
- `sc.pp.log1p` sets `adata.uns['log1p']`, so a second call in the same session
  warns but still applies.
- Because the shipped checkpoints were trained under one convention and the
  current scripts apply another, it is not currently clear from the repo alone
  which convention `checkpoint/*.pt` expects. **Verify against a known-good
  reference before trusting fine-tuned results.**

Note also that `normalize_total` is called with no `target_sum`, so it
normalises to the median library size of whatever subset is loaded — making the
scaling dataset-dependent rather than fixed.

**Fix would change:** the input distribution the model sees, i.e. everything.
Resolving this needs a decision about what the checkpoints were trained on, not
just a code edit.

---

## Also worth knowing (not behaviour-changing, not fixed here)

These are safe to fix at any time; they were left out only to keep the
compatibility commit narrow.

- **`DrugDoseAnnDataset.__getitem__` is O(n) per sample.** It calls
  `self.dense_adata_index.index(self.paired_control_index[index])`, a linear
  scan of a Python list, once per item. On large datasets this dominates data
  loading. A dict from label to position, built once in `__init__`, is a
  drop-in replacement.
- **Result files are opened in append mode.** `test()` writes with `'a'`/`'a+'`
  and prints a warning about it. Repeated runs silently concatenate, after
  which `analysis_*.py` reads mismatched row counts. Clear the results
  directory between runs.
- **Checkpoint filenames can collide with the pretrained weights.** Training
  saves to `model_save_dir + split_key + '_best_epoch_all.pt'`. A `split_key`
  of `lincs` or `sciplex` overwrites the shipped checkpoint of the same name.
- **`test_demo.py` ignores its own `--save_dir`** and hardcodes
  `Trainer.test('./checkpoint/lincs_best_epoch_all.pt')`, while taking
  `x_dimension` from the data. Non-978-gene input fails on a shape mismatch
  rather than a clear message.
- **`data/_utils.py:Condition_encoder` uses `OneHotEncoder(sparse=False)`**,
  removed in scikit-learn 1.4 (renamed `sparse_output`). The function is
  currently unused, so `requirements.txt` pins `scikit-learn==1.3.2` rather
  than editing dead code.
- **`AllChem.GetMorganFingerprintAsBitVect`** is deprecated in RDKit 2024.09+
  in favour of `MorganGenerator`; `requirements.txt` pins an older RDKit.
- **No fine-tuning path exists.** `train()` always constructs a fresh `PRnet`
  and applies `weight_init`; only `test()` / `get_latent()` / `get_per_latent()`
  load a checkpoint.
