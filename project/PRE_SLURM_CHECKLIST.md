# Before every `sbatch` (local preflight)

HPC jobs are expensive. Run these checks **on your laptop or login node** before syncing and submitting.

## 1. One command (recommended)

From the `project/` directory:

```bash
cd "/path/to/Max Planck Project/project"
pip install timm tifffile  # once, same deps as Slurm venv
python validate_before_slurm.py
```

With your real data (stronger check — loads TIFFs + CSVs):

```bash
python validate_before_slurm.py \
  --data-root "data/Max Planck Data/Gold Particle Labelling/analyzed synapses"
```

Or use the wrapper:

```bash
./hpc/validate_before_submit.sh --data-root "data/Max Planck Data/Gold Particle Labelling/analyzed synapses"
```

**Exit code 0** → imports, model forward, loss, optional data batch, and `bash -n` on `hpc/*.slurm` all succeeded.

## 2. What this catches

| Failure type | Caught by |
|--------------|-----------|
| Missing `timm` / broken imports | Step 1 |
| Shape / loss bugs (e.g. 64 vs 256) | Step 2 (synthetic batch) |
| Syntax errors in training scripts | Step 3 (`py_compile`) |
| Wrong paths / unreadable TIFFs | Step 4 (if `--data-root` set) |
| Broken Slurm shell (bad line endings, typos) | Step 5 (`bash -n`) |

## 3. What it does **not** replace

- **Cluster-only** issues: module versions, CUDA driver, queue limits, wall time.
- **Full training** — this is a **few-second** smoke test.

## 4. Optional: hook into submit script

```bash
VALIDATE=1 ./hpc/submit_centernet_training_athene.sh   # if you add a check to the script
```

(You can add `validate_before_slurm.py` at the top of your submit script manually.)

---

See also: `RESEARCH_PROTOCOL.md` for evaluation rigor.
