# Environment and provenance

Use Python 3.12. The source uses `hashlib.file_digest` and therefore needs at least Python 3.11. `requirements.txt` matches the local environment used for the export's CPU checks: PyTorch 2.6.0+cu124, torchvision 0.21.0+cu124, NumPy 2.4.6, Pillow 12.3.0, SciPy 1.15.3, clean-fid 0.1.35, kagglehub 1.0.2 and tqdm 4.70.1. Select PyTorch wheels appropriate to your hardware; a CUDA wheel can still run CPU inference.

The original portable archive's requirements are preserved at [reference/original_requirements.txt](reference/original_requirements.txt). They record an earlier environment with PyTorch 2.7.1 and NumPy 2.2.6. They are provenance, not the environment used for this export's checks.

`EXPORT_PROVENANCE.json` identifies every original file copied byte-for-byte. The original `provenance.json` is also retained because training and data preparation enforce its hashes for protected course files, split manifests and backbone code. `reference/cache_identity.json` records the expected decoded training-image and class-label hashes. If data preparation reports a mismatch, investigate the dataset, transparency conversion and decoder versions; do not remove the integrity checks to silently accept different pixels.

Core training, model, dataset and evaluation files are unchanged. `best_sampler.py` and `sample.py` provide the selected nonuniform inference schedule without changing the course `Model` parent class or default sampler. No checkpoint, dataset image, cache, training log or generated image is included.

The export checks cover imports, CLI help, existing CPU unit tests and strict loading of the original 101k EMA checkpoint. They do not constitute a new training run, GPU memory benchmark or FID reproduction. Historical training/evaluation measurements are documented in the repository's top-level README and data/ directory.

Upstream attribution and the CC BY-NC-SA 4.0 license are retained in [src/original_meanflow/UPSTREAM.md](src/original_meanflow/UPSTREAM.md) and [src/original_meanflow/LICENSE](src/original_meanflow/LICENSE).
