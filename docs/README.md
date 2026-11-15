# Project page and recorded experiments

This directory is a static GitHub Pages site. It needs no package manager, build service, analytics, or external scripts. `index.html`, `style.css`, and `app.js` load only local assets. Serve it with:

```bash
python3 -m http.server 8000 --directory docs
```

GitHub Pages can serve `main` → `/docs`. `.nojekyll` keeps these files unchanged. Repository visibility and Pages settings are separate from the source files.

## What the images show

- `assets/gallery/`: 12 declared species × 16 seeds: 7, 19, 42, 73, 101, 256, 512, 777, 999, 1234, 1337, 2026, 2048, 4096, 8192, 16384. Each strip contains initial Gaussian noise and four actual model states from the 101k EMA checkpoint with times `[1, .6, .3, .018, 0]`. Intermediate states are clipped for display. These are CPU FP32 samples, not members of the GPU FID evaluation set.
- `assets/study/compare-*`: the same 12 species and three seeds at 64k and 101k, using uniform four-call sampling on the GPU. All outputs are included. Grid order follows the species list in `study.json`, left to right, top to bottom.
- `assets/study/journey-charmander-*`: fixed Charmander class and seeds 7, 19, 42, 73, 101, 256, 512, and 777 at all 168 evaluated, retained EMA checkpoints, using uniform four-call sampling. The older Pikachu frames remain available under `journey-*` with `training-frames-pikachu.json`; their six original snapshots were independently reproduced pixel for pixel. `training-frames.json` records every checkpoint and image hash. `training-eight-atlas.png` packs all 1,344 images losslessly into 32 columns for playback. The older seed-42 sheet is retained as `training-atlas.png`.
- `assets/four-calls.gif`: a preview rendered from six seed-42 gallery strips. It contains only the five saved model states. There is no interpolation.
- The opening canvas renders the recorded final images as colored characters. It holds each Pokémon for 3.5 seconds, then uses a 1.2-second fade-out/fade-in to reveal the next one. These transitions are presentation between recorded outputs, not model inference. Click the artwork to pause or resume; with reduced motion enabled it stays still and clicking advances manually. The three small image buttons select Mew, Gengar, or Dragonite. Rotation stops when the opening is off-screen or the tab is hidden.

The gallery displays five consecutive entries from the 16 recorded seeds, wrapping at the end. Each image can replay individually; “Replay all five” plays them together. The seed slider selects the first entry. Images default to pixel rendering and stay at 128 CSS pixels or smaller in the gallery (64 pixels on phones). The checkpoint wipe opens on Bulbasaur and uses all 36 existing matched pairs; it does not mix samples from different seeds. The training view puts a compact FID chart on the left and eight recorded Charmanders on the right. All eight seeds advance together at each of the 168 measured FID points, always labeling that checkpoint’s update count. On small screens the images come first. Playback uses the local image sheet and advances every 110 ms, so image loading cannot stall individual frames. On desktop, moving over the opening’s empty space leaves small recorded samples in place; each fades after 3.5 seconds. This trail is confined to the opening, bounded to ten images, and disabled for touch and reduced-motion preferences. No Pokémon follow the cursor around the page. The trail has no header control.

The samples are conditioned on the class names in the course mapping. They are not selected for image quality. Their declared species list is a small qualitative selection, not a random sample of all 151 classes or a human-preference benchmark.

## Regenerate the gallery

Use the repository's documented model dependencies, plus Pillow. The exporter checks the checkpoint hash, strictly loads the state dict, and checks that its first trajectory exactly matches `sample_best_four` before writing images.

```bash
python scripts/export_gallery.py \
  --checkpoint code/checkpoints/ema_0101000.ckpt \
  --output outputs/gallery-recheck --device cpu \
  --seeds 7 19 42 73 101 256 512 777 999 1234 1337 2026 2048 4096 8192 16384
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

Original generated PNGs contain RGB pixels without EXIF, text, or location metadata. The presentation copies contain the same RGB bytes plus an alpha mask; the originals remain available through “Show backgrounds”. Checkpoint files contain model tensors plus objective, step, and guidance metadata. Raw server logs, hostnames, credentials, dataset images, and optimizer states are not included in the new page or releases. Publication checks cover local assets and checkpoint hashes; they cannot establish that a trained model has never memorized any training image.

The current provenance documents omit local filesystem paths while retaining the original source hashes. Earlier Git commits and the older inference-source release contain historical local path strings; history has not been rewritten.

The page uses system sans-serif fonts and native MathML for the sampling equation. The training-loop diagram uses responsive HTML stages, with click-to-explain and optional playback. No external math or animation library is needed.

The optional text-art canvas uses IBM Plex Mono; the earlier EB Garamond asset is retained for provenance but is no longer loaded by the page. Both bundled fonts come from the official `google/fonts` repository and retain their SIL Open Font License files. The source download was checked against GitHub's blob hashes:

| Asset | Git blob SHA-1 |
| --- | --- |
| EB Garamond variable TTF | `52d34b86ec20977f222ce37e8158e6b3002c0a31` |
| IBM Plex Mono Light TTF | `61d43577842179cdffeb38b6c1928786981db0da` |

Design references: [antwi.dev](https://antwi.dev/), [jadefranson.com](https://www.jadefranson.com/), [Visual Anagrams](https://dangeng.github.io/visual_anagrams/), and [Prompt-to-Prompt](https://prompt-to-prompt.github.io/). Their images, source code, and site assets were not copied. All Pokémon images on this page come from this project's model. The model backbone's upstream license and attribution remain in `code/src/original_meanflow/`.

## Optional Mew sound

Clicking the “Pokémon!” wordmark plays the short legacy Mew cry at 35% volume, only after a click. It also selects Mew in the artwork. Audio is never started by the automatic rotation and pauses if the tab is hidden. The 8,552-byte Ogg file is served locally.

The source is the `PokeAPI/cries` repository linked by PokéAPI’s Pokémon-cries documentation, pinned to commit `b3a9044bb839b5e3152aaa536c7b463a0fcf0140`. The audio and upstream notice were checked against GitHub’s blob hashes before inclusion. `assets/audio/provenance.json` records source URLs and hashes. The upstream notice identifies audio copyright as The Pokémon Company; the repository’s CC0 notice does not remove that audio copyright. No third-party code was downloaded or executed.

## Dense training replay

Run the exporter against the already-retained training run and matching published source. It strictly loads every EMA state dict, validates checkpoint steps and objective, fixes Charmander class 14 and eight predeclared seeds, and uses FP32 with TF32 disabled. It never downloads or changes training files.

```bash
python scripts/export_training_frames.py \
  --source-root /path/to/matching-source \
  --run-root /path/to/training-run \
  --output /path/to/fresh-training-frames --device cuda
```

The selected steps exactly match `assets/study/fid_history.csv`: 1,000-update intervals early, 250-update intervals through much of the fine-tuning stage, and the available later checkpoints. The same eight seeds keep their positions across the entire run, and all 1,344 outputs are included. There are no interpolated model weights or images. Rebuild the browser data and lossless image sheet with `python scripts/build_page_data.py`.

## Transparent presentation images

`assets/cutouts/` contains display copies for all gallery samples, sampling strips, matched comparisons, and training frames. BiRefNet-general predicts an alpha mask using rembg-compatible preprocessing and ONNX Runtime 1.22.0 on the GPU. For flat pale backgrounds, an edge-connected color mask supplements the neural prediction while protecting foreground interiors and removing small, unconfident border artifacts. Initial Gaussian noise is kept whole because it has no foreground object.

This process does not repaint or recolor the images: every source RGB byte is retained and only alpha is added. A mask can still change the apparent outline; it is presentation, not generator output or an evaluation preprocessing step. FID measurements use the untouched full images. The two “Show backgrounds” controls switch the whole page back to the original frames.

The official rembg and BiRefNet research repositories were checked before use. The model came from rembg’s official release, was verified against its published MD5, and has SHA-256 `58f621f00f5d756097615970a88a791584600dcf7c45b18a0a6267535a1ebd3c`. Model weights are not bundled with the page. The 98 reviewed polygon outlines in `data/display-mask-outlines.json` correct difficult backgrounds in gallery, comparison, and training images. They are keyed to the source RGB hash and apply consistently wherever the same image appears, including image sheets and matching sampling-strip cells. For 166 other difficult training foregrounds, `data/display-mask-guides.json` links each source RGB hash to a reviewed outline. OpenCV 5.0.0 transfers that alpha outline using Farneback optical flow at 256 × 256, reduces it to native resolution, allows a one-pixel edge margin, and intersects it with the neural mask. Only alpha is transformed; source RGB pixels are never warped. The builder requires a predownloaded, matching `BiRefNet-general-epoch_244.onnx` and refuses to run if its checksum differs.

```bash
python -m pip install 'onnxruntime==1.22.1' 'numpy==2.2.6' 'pillow==11.3.0' 'scipy==1.15.3' 'opencv-python-headless==5.0.0.93'
python scripts/build_cutouts.py --model /path/to/BiRefNet-general-epoch_244.onnx
python scripts/build_preview_animation.py
```

The example uses CPU ONNX Runtime; GPU generation uses ONNX Runtime GPU 1.22.0, the existing compatible PyTorch CUDA/cuDNN libraries, and `--device cuda`. Use a fresh output directory when rebuilding masks. `assets/cutouts/manifest.json` records the method, model checksum, and generated display files.
