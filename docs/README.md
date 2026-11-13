# Project page and recorded experiments

This directory is a static GitHub Pages site. It needs no package manager, build service, analytics, or external scripts. `index.html`, `style.css`, and `app.js` load only local assets. Serve it with:

```bash
python3 -m http.server 8000 --directory docs
```

GitHub Pages can serve `main` → `/docs`. `.nojekyll` keeps these files unchanged. Repository visibility and Pages settings are separate from the source files.

## What the images show

- `assets/gallery/`: 12 declared species × seeds 42, 1234, 2026. Each strip contains initial Gaussian noise and four actual model states from the 101k EMA checkpoint with times `[1, .6, .3, .018, 0]`. Intermediate states are clipped for display. These are CPU FP32 samples, not members of the GPU FID evaluation set.
- `assets/study/compare-*`: the same 12 species and three seeds at 64k and 101k, using uniform four-call sampling on the GPU. All outputs are included. Grid order follows the species list in `study.json`, left to right, top to bottom.
- `assets/study/journey-*`: fixed Pikachu class and seed 42 at 1k, 8k, 32k, 64k, 101k, and 122k EMA checkpoints, using uniform four-call sampling.
- `assets/four-calls.gif`: a preview rendered from six seed-42 gallery strips. It contains only the five saved model states. There is no interpolation.
- The opening canvas converts those same pixels into colored characters. Its “type / pixels” control switches to the recorded RGB image. This typographic treatment is presentation, not another model output.

The samples are conditioned on the class names in the course mapping. They are not selected for image quality. Their declared species list is a small qualitative selection, not a random sample of all 151 classes or a human-preference benchmark.

## Regenerate the gallery

Use the repository's documented model dependencies, plus Pillow. The exporter checks the checkpoint hash, strictly loads the state dict, and checks that its first trajectory exactly matches `sample_best_four` before writing images.

```bash
python scripts/export_gallery.py \
  --checkpoint code/checkpoints/ema_0101000.ckpt \
  --output outputs/gallery-recheck --device cpu
```

The output must be a fresh directory. To rebuild the browser data and README animation from the checked-in assets:

```bash
python scripts/build_page_data.py
python scripts/build_preview_animation.py
node --check docs/app.js
```

## Reproduce the GPU study

`scripts/checkpoint_study.py` uses the original implementation, an existing run directory, and the original dataset. It never downloads assets. It expects the already-cached clean-fid Inception TorchScript weights at `/tmp/inception-2015-12-05.pt`; the exact hash is recorded in `assets/study/study.json`.

```bash
python scripts/checkpoint_study.py \
  --source-root code \
  --run-root /path/to/training-run \
  --data-root code/data/pokemon-generation-one-22k \
  --output /path/to/fresh-study

python scripts/baseline_sensitivity.py \
  --study /path/to/fresh-study \
  --data-root code/data/pokemon-generation-one-22k
```

The [released checkpoints](../data/checkpoints.json) cover 1k, 8k, 32k, 64k, and 101k. The 122k snapshot image is included, but its weight file is not included in these downloads. Reproducing the full six-image sequence also requires that original checkpoint or a new training run.

The run directory must contain the six displayed checkpoints, `metrics.jsonl`, `fid.csv`, `status.json`, and the evaluator's `official_validation_reference_png/`. The dataset must include the original images as well as the fixed manifests. The script checks the model source hash. The full study requires a CUDA device and sufficient working space for temporary PNGs and Inception features.

`publication/` contains the page's images and numeric measurements. `private_analysis/` contains intermediate real images, features, and generated FID sets; it is deliberately excluded from the repository. Never copy the raw run directory into the page.

## Reading the measurements

The comparison reevaluates 3,020 generated samples, 20 per class, against the 3,020 course validation images. Generation uses seed 1234, batch size 32, FP32, TF32 disabled, and the existing course PNG/clean-fid pipeline. The remeasured FIDs match the recorded values to below 0.000001.

The baseline experiment first compares all 17,079 training images to all 3,020 validation images. The sensitivity check then reuses those exact feature arrays, drawing 20 training images per class without replacement with NumPy `default_rng` seeds 42, 1234, 2026. Class names are sorted alphabetically. Both the sample count and class composition change relative to the full training split. This is not an isolated sample-count experiment or an exact reconstruction of the course's baseline protocol.

The loss CSV exports only `step`, `loss`, and cumulative `training_seconds`, retaining the last logged occurrence of each step. The page summarizes these in 1,000-update bins with mean, minimum, and maximum. The shaded range is not a confidence interval. The original report's older history remains in the root `data/` directory.

## Assets, privacy, and attribution

Generated PNGs contain RGB pixels without EXIF, text, or location metadata. Checkpoint files contain model tensors plus objective, step, and guidance metadata. Raw server logs, hostnames, credentials, dataset images, and optimizer states are not included in the new page or releases. Publication checks cover local assets and checkpoint hashes; they cannot establish that a trained model has never memorized any training image.

The current provenance documents omit local filesystem paths while retaining the original source hashes. Earlier Git commits and the older inference-source release contain historical local path strings; history has not been rewritten.

Fonts come from the official `google/fonts` repository and retain their SIL Open Font License files. The source download was checked against GitHub's blob hashes:

| Asset | Git blob SHA-1 |
| --- | --- |
| EB Garamond variable TTF | `52d34b86ec20977f222ce37e8158e6b3002c0a31` |
| IBM Plex Mono Light TTF | `61d43577842179cdffeb38b6c1928786981db0da` |

Design references: [antwi.dev](https://antwi.dev/) and [jadefranson.com](https://www.jadefranson.com/). Their images, source code, and site assets were not copied. All Pokémon images on this page come from this project's model. The model backbone's upstream license and attribution remain in `code/src/original_meanflow/`.
