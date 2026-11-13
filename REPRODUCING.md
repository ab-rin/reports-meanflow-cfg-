# Running the model

The implementation is in [code/](code/). Its [README](code/README.md) covers data preparation, training, resume, sampling, and the existing CPU checks.

## Checkpoint

Download [ema_0101000.ckpt](https://github.com/ab-rin/reports-meanflow-cfg-/releases/download/meanflow-cfg-101k/ema_0101000.ckpt) and place it in `code/checkpoints/`. Access to this private repository requires GitHub authentication. With the GitHub CLI, run from the repository root:

```bash
gh release download meanflow-cfg-101k \
  --repo ab-rin/reports-meanflow-cfg- \
  --pattern ema_0101000.ckpt --dir code/checkpoints
```

```text
Filename: ema_0101000.ckpt
Bytes: 224175708
SHA256: ab99f610b9bc2a8ec26f627573608b52d490065773ede28ab18bf5a940d11b41
```

This is the original EMA checkpoint at update 101,000. The sampler settings are separate from its weights.

## Selected four-step sampler

Run from `code/` after installing its dependencies:

```bash
python sample.py \
  --checkpoint checkpoints/ema_0101000.ckpt \
  --output outputs/example --schedule best4 \
  --classes 0 25 150 --samples-per-class 2 --device cuda
```

Or load it in Python:

```python
import torch
from model import Model
from best_sampler import sample_best_four

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.manual_seed(1234)

model = Model.load_checkpoint(
    "checkpoints/ema_0101000.ckpt",
    evaluate_mode="few_nfe",
    device="cuda",
)
categories = torch.tensor([0, 1], device="cuda")
images = sample_best_four(
    model, (2, 3, 64, 64), device="cuda", category=categories,
)
```

`sample_best_four` uses `[1, 0.6, 0.3, 0.018, 0]`, makes exactly four model calls, and returns RGB tensors in `[-1, 1]`. The class IDs come from [category_to_id.json](code/data/pokemon-generation-one-22k/category_to_id.json), not Pokédex numbers. Guidance scale 2 is already learned in the conditional model.

The original `ModelFewNFE.sample()` and course evaluator retain their uniform-interval default. Calling them directly does not select the tuned schedule. This preserves the meaning of the original training logs and the provided evaluator.

## Historical ρ = 2 evaluation

The nonuniform markers in the training-history figure use the earlier power schedule `[1, 0.5625, 0.25, 0.0625, 0]`; the dense curve uses uniform intervals. After preparing the dataset, the power schedule's seed-1234 evaluation can be run from `code/`:

```bash
python sweep_sampling_schedule.py \
  --checkpoint checkpoints/ema_0101000.ckpt \
  --data-root data/pokemon-generation-one-22k \
  --output outputs/selected_101k_rho2 \
  --reference official_validation \
  --rhos 2 --seed 1234 --samples-per-class 20 \
  --batch-size 32 --fid-batch-size 32
```

This script evaluates power schedules; it does not express the later tuned schedule. The tuned schedule's recorded confirmation data are in [data/best_schedule.json](data/best_schedule.json).

## Evaluation conditions

All reported FIDs use 3,020 generated images, 20 per class, against all 3,020 official validation images. Generation uses batch size 32, FP32, TF32 disabled, and the original PNG conversion and clean-fid pipeline. The recorded environment was Python 3.12.3, PyTorch 2.6.0+cu124, torchvision 0.21.0+cu124, NumPy 2.4.6, SciPy 1.15.3, Pillow 12.3.0, and clean-fid 0.1.35. The exported historical requirements describe the earlier training environment; see the code README for the distinction.

Changing software, precision, batch size, preprocessing, or seed can change FID. The short loading example verifies how to use the selected sampler; it does not reproduce the full evaluation by itself.

## Training-time accounting

[data/run_history.json](data/run_history.json) distinguishes active update time from process elapsed time. Time at a selected checkpoint is cumulative from initialization, not the duration of its last resumed session. Repeated steps after recovery are not counted twice in the reconstructed trajectory.

The original report ended at a valid 110,600-update checkpoint. A later continuation, inspected for the project page, reached 122,295 retained updates; the last recorded evaluation was at 122,000. The selected checkpoint remains 101,000. The original data files preserve the earlier report snapshot; [the page data](docs/assets/study/) contains the extended history and new measurements. See [the page reproduction notes](docs/README.md) for the matched checkpoint comparison and real-data baseline checks.
