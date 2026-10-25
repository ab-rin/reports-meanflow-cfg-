# Reproducing the selected sampler

The selected model uses the existing course directory layout. The following commands run from the MeanFlow CFG implementation root, where `model.py`, `evaluate.py`, `sweep_sampling_schedule.py`, and `src/` are located. In the original workspace this directory is `meanflow_cfg_implementation_20260929/`.

## Checkpoint and environment

Use the EMA checkpoint from update 101,000:

```text
Filename: ema_0101000.ckpt
Bytes: 224175708
SHA256: ab99f610b9bc2a8ec26f627573608b52d490065773ede28ab18bf5a940d11b41
Original path: runs/meanflow_cfg_100k/checkpoints/ema_0101000.ckpt
```

The checkpoint is approximately 214 MiB and is not part of the report. Copy an existing verified copy to `checkpoints/ema_0101000.ckpt` before running these commands. No replacement checkpoint is downloaded automatically.

The selected evaluation recorded Python 3.12.3, PyTorch 2.6.0+cu124, torchvision 0.21.0+cu124, NumPy 2.4.6, SciPy 1.15.3, Pillow 12.3.0, and clean-fid 0.1.35. The historical export's `requirements.txt` describes the earlier environment; it is not a record of this local evaluation. Use course-approved dependencies for submission.

## Load the exact four-call configuration

```python
from model import Model

model = Model.load_checkpoint(
    "checkpoints/ema_0101000.ckpt",
    evaluate_mode="few_nfe",
    sampling_rho=2.0,
    device="cuda",
)
```

Pass `sampling_rho=2.0` explicitly. The historical model default is `1.0`, which uses uniform intervals and is also used by periodic training evaluation. Leaving that default unchanged preserves the meaning of the original training logs.

## Full validation FID

Prepare the dataset with the unchanged course loader and original split files. Then use a fresh output directory:

```bash
python sweep_sampling_schedule.py \
  --checkpoint checkpoints/ema_0101000.ckpt \
  --data-root data/pokemon-generation-one-22k \
  --output outputs/selected_101k_rho2 \
  --reference official_validation \
  --rhos 2 \
  --seed 1234 \
  --samples-per-class 20 \
  --batch-size 32 \
  --fid-batch-size 32
```

The script verifies protected source hashes, loads weights strictly, counts backbone calls, and records the schedule and reference set. For 3,020 images at batch size 32, its generation loop should report 380 backbone calls over 95 batches. Warm-up calls are outside that recorded generation loop.

Changing software, precision, batch size, preprocessing, or random seed can change FID. The README reports the saved measurement rather than promising bitwise agreement on another machine.

## Training-time accounting

`data/run_history.json` distinguishes active update time from process elapsed time and identifies the original logs. Time at a selected checkpoint is cumulative from initialization, not the duration of its latest resumed session. Repeated steps after a recovery are not counted twice in the reconstructed baseline trajectory.

The local CFG run has a valid saved checkpoint at 110,600 updates. A log line at 110,700 precedes a failed save; the configured target of 125,000 was not completed. The selected checkpoint remains 101,000.
