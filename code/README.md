# MeanFlow CFG training and inference

This directory contains the original 56,010,243-parameter, 151-class, 64×64 RGB MeanFlow CFG implementation, plus a portable helper for the selected four-call schedule `[1, 0.6, 0.3, 0.018, 0]`. The original model, training objective and course evaluation files are unchanged. Weights and images are not included.

Run commands from this directory:

```bash
cd code
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

See [ENVIRONMENT.md](ENVIRONMENT.md) for version and source provenance details. CUDA training requires compatible PyTorch wheels and a GPU; the documented CPU tests and inference example can run without one.

## Generate images

Supply an existing compatible EMA checkpoint. For the measured checkpoint, the filename is `ema_0101000.ckpt` and SHA256 is `ab99f610b9bc2a8ec26f627573608b52d490065773ede28ab18bf5a940d11b41`.

```bash
python sample.py --checkpoint /path/to/ema_0101000.ckpt \
  --output outputs/example --schedule best4 \
  --classes 0 25 150 --samples-per-class 2 --device cpu
```

Use `--device cuda` for GPU inference. Category IDs come from [category_to_id.json](data/pokemon-generation-one-22k/category_to_id.json), not Pokédex numbers. Omitting `--classes` and `--samples-per-class` generates the course-balanced 3,020-image set. `sample.py` saves PNGs and sampling metadata in a fresh output directory; it does not compute FID. It also supports `uniform4`, `rho2-4` and `one` schedules for baseline comparisons.

To load and sample in Python:

```python
import torch
from model import Model
from best_sampler import sample_best_four

model = Model.load_checkpoint(
    "/path/to/ema_0101000.ckpt", evaluate_mode="few_nfe", device="cpu"
).eval()
images = sample_best_four(
    model, (1, 3, 64, 64), device="cpu", category=torch.tensor([0])
)
```

The helper makes exactly four conditional network calls and returns tensors in `[-1, 1]`. Guidance is learned during training, so sampling needs no extra unconditional pass. The unchanged `model.sample()` defaults to uniform intervals; use the helper for the selected schedule. Reusing these weights with one call does not make them a distilled one-step model.

## Prepare data and train from scratch

Fixed split manifests and the class mapping are included; dataset pixels are downloaded separately. The downloader composites transparency onto white, matching the course preprocessing.

```bash
python download_dataset.py --data_root data/pokemon-generation-one-22k
python prepare_data.py --dev-per-class 0 \
  --fid-reference official_validation --seed 20260923

CUDA_VISIBLE_DEVICES=0 python train.py \
  --output outputs/meanflow_cfg --updates 101000 \
  --microbatch 16 --effective-batch 512 \
  --guidance-scale 2 --label-dropout 0.1 \
  --schedule fastgen --seed 42 \
  --fid-every 1000 --fid-batch 32 \
  --save-every 100 --preview-every 100 --log-every 10 --compile
```

The `fastgen` preset uses Adam at `3e-4`, 500 warmup updates and EMA decay `0.999`. Training starts from random initialization and uses only the fixed official training split. `--updates` is the total target, not additional updates. The unchanged trainer measures peak CUDA allocation and enforces its 20 GB limit. Microbatch choice and compilation affect memory and throughput.

Resume by repeating the same command with `--resume outputs/meanflow_cfg/training_latest.pt` and a larger total `--updates` if needed. Resume requires the full training state, including optimizer and RNG state; the EMA inference checkpoint alone is insufficient. The run writes `ema_latest.ckpt`, full resumable checkpoints and periodic EMA snapshots under the output directory.

## Evaluate and verify

`evaluate.py` is the unchanged course evaluator, and its `few_nfe` default uses uniform intervals. To reproduce the historical rho=2 sampler, run:

```bash
python sweep_sampling_schedule.py \
  --checkpoint /path/to/ema_0101000.ckpt --output outputs/rho2_evaluation \
  --data-root data/pokemon-generation-one-22k --reference official_validation \
  --rhos 2 --seed 1234 --samples-per-class 20 --batch-size 32 --fid-batch-size 32
```

For the selected explicit schedule, first generate its balanced PNG set:

```bash
python sample.py --checkpoint /path/to/ema_0101000.ckpt \
  --output outputs/best4_evaluation --schedule best4 --device cuda
```

Then use the unchanged course reference preparation and FID functions:

```python
import torch
from dataset import PokemonDataModule
from evaluate import prepare_reference_set, compute_fid

torch.set_float32_matmul_precision("highest")
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
data = PokemonDataModule(
    data_root="data/pokemon-generation-one-22k",
    split_dir="data/pokemon-generation-one-22k", batch_size=32, num_workers=4,
)
reference = prepare_reference_set(data, "outputs/official_validation_reference")
print(compute_fid("outputs/best4_evaluation", reference, device="cuda", batch_size=32))
```

Use fresh output/reference directories and preserve the fixed class balance, preprocessing and reference identity. These commands execute a full evaluation; the export checks did not rerun it.

Run the included original CPU tests:

```bash
OMP_NUM_THREADS=4 python -m unittest discover -s tests -v
```

They check the guided objective/JVP, null-class conditioning, sampling schedules/call counts and full-state resume settings. All 25 tests passed during export; loading and CLI checks are recorded in [VALIDATION.json](VALIDATION.json). Original license and attribution are in [src/original_meanflow](src/original_meanflow/). Export provenance is in [EXPORT_PROVENANCE.json](EXPORT_PROVENANCE.json).
