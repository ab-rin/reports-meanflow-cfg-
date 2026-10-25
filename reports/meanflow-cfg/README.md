# MeanFlow CFG: four-call generation

Our best recorded configuration is the **101,000-update EMA checkpoint**, sampled with the schedule **1 → 0.5625 → 0.25 → 0.0625 → 0**. It reaches **19.8764 FID** on the 3,020-image course validation set. The same weights with uniform intervals score 20.3921.

This report covers four-call generation only. Results are local measurements using the course evaluator, rather than an official TA score. The selected checkpoint and schedule were chosen using validation FID.

## What we trained

| Setting | Value |
|:--|:--|
| Task | Class-conditional Pokémon generation, 64 × 64 RGB |
| Data | 17,079 training images; 151 classes; 3,020 validation images |
| Model | Pixel-space MeanFlow U-Net, 56,010,243 parameters |
| Objective | Original MeanFlow with built-in classifier-free guidance |
| Guidance | Fixed scale 2.0; label dropout 0.1 during training |
| Initialization | Random initialization, seed 42; no external pretrained generator |
| Optimizer | Adam, learning rate 0.0003; 500-update warmup |
| EMA | Decay 0.999 |
| Effective batch | 512 images per optimizer update |
| GPU microbatch | 16 initially, then 32 with gradient accumulation adjusted |
| Hardware | One NVIDIA GRID A100D-20C, 20 GB |
| Peak training allocation | 12.43 GB recorded by PyTorch |
| Software | Python 3.12.3, PyTorch 2.6.0+cu124, torchvision 0.21.0+cu124 |

Guidance is learned during training. Sampling uses the conditional network once per interval; it does not add an unconditional guidance pass. The source file `imf_objective.py` supplies a shared JVP helper, but this run uses the MeanFlow CFG objective in `src/cfg_meanflow.py`.

## The four calls

![The four model calls](assets/four_calls.svg)

Each call predicts the average velocity over one interval:

```text
u = backbone(x, t, r, category)
x = x - (t - r) * u
```

| Call | Start t | End r | Update size t − r |
|:--:|--:|--:|--:|
| 1 | 1.0000 | 0.5625 | 0.4375 |
| 2 | 0.5625 | 0.2500 | 0.3125 |
| 3 | 0.2500 | 0.0625 | 0.1875 |
| 4 | 0.0625 | 0.0000 | 0.0625 |

The times follow `t[i] = (1 - i / 4) ** 2`. All four calls share the same weights and category. Sampling starts with Gaussian noise and clamps the final pixels to `[-1, 1]`. There is no learned decoder, refiner, or extra denoising call. The recorded evaluation counted **380 backbone calls across 95 batches: exactly four per batch**.

## Results and training time

| Run | Training hardware | Retained updates | Selected update | Time to selected | Total active time | FID ↓ | Schedule |
|:--|:--|--:|--:|--:|--:|--:|:--|
| Original MeanFlow, no CFG | 1 × A100 20 GB | 26,320 | 26,000 | 52.08 h | 52.65 h | 26.1075 | Uniform |
| Earlier MeanFlow CFG | RTX PRO 6000 + 7 × RTX 3090 | 70,000 | 62,000 | Not retained | Not retained | 21.4730 | ρ = 2 |
| **Selected MeanFlow CFG** | **1 × A100 20 GB** | **110,600** | **101,000** | **268.24 h** | **290.79 h** | **19.8764** | **ρ = 2** |

All three runs use 56,010,243 parameters and effective batch 512. The older eight-GPU run is included as historical context; its FID is a local rescore of the archived weights. Its training duration was not retained. The local CFG target was 125,000 updates, but the last valid saved state is 110,600.

Active training time excludes FID evaluation, checkpoint writing, and time while the process was stopped. These runs use different training budgets and hardware, so the table records experiment history rather than a controlled comparison of algorithms.

![Four-call validation FID during training](assets/fid_history.svg)

The dense curve uses uniform intervals. The three marked nonuniform evaluations use the selected power schedule on the same local run. The older 62k checkpoint belongs to a separate run and is excluded from this curve.

## Evaluation details

We generated 20 images per class, for 3,020 images total, with seed 1234 and batch size 32. FID uses all 3,020 course validation images. Evaluation ran in FP32 with TF32 disabled. Training used FP32 with TF32 enabled.

At checkpoint 101k, changing only the schedule reduced FID by **0.5156**, or **2.53%**. This is a result for the recorded seed and evaluation setup; it is not an average over repeated seeds.

Generation and PNG export took **71.28 seconds** (about **42.4 images/second**). Including FID computation took **89.03 seconds**. These are batch-throughput measurements, not single-image latency. Recorded peak allocation during this evaluation was 2.304 GB.

The selected checkpoint is `ema_0101000.ckpt` (224,175,708 bytes). Its identity and the underlying measurements are in [data/selected_result.json](data/selected_result.json). Model weights and generated image collections are not embedded in this report.

## Files and provenance

- [data/run_history.json](data/run_history.json): training budgets, timing definitions, and original evidence paths.
- [data/fid_history.csv](data/fid_history.csv): the four-call observations used in the graph.
- [data/meanflow_cfg_nfe4_rho2.csv](data/meanflow_cfg_nfe4_rho2.csv): the three measured nonuniform-schedule results.
- [data/comparison.csv](data/comparison.csv): the validation scores compared in the table.
- [data/selected_result.json](data/selected_result.json): selected checkpoint, schedule, environment, and evaluation.
- [Reproduction notes](REPRODUCING.md): how to select the sampler explicitly without changing course files.
- [plot_figures.py](plot_figures.py): redraws the figures from the included measurements.

All tables, graphs, and their supporting data live in this report folder. The original experiment paths in the JSON files are relative to the local FastGen project, so each measurement can be traced back to its source.

Method: [Mean Flows for One-step Generative Modeling](https://arxiv.org/abs/2505.13447), including built-in guidance. Backbone source: [Gsunshine/py-meanflow](https://github.com/Gsunshine/py-meanflow/tree/1f6d72d94247c8fdeb89489acca1a8007a6baf6c), licensed CC BY-NC-SA 4.0. Task and evaluator: [KAIST FastGen Challenge](https://generative-models.kaist.ac.kr/fastgen/).
