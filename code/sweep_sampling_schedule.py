"""Compare original/CFG MeanFlow schedules against official validation by default.

An explicit training-reference mode is also available. The reference and all
schedule settings are recorded, and no checkpoint is changed.
"""

import argparse
import csv
import json
import math
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from dataset import PokemonDataset, read_split
from evaluate import compute_fid, generate_samples, prepare_reference_set
from model import Model, ModelFewNFE
from src.runtime import assert_protected, evaluator_precision, isolated_seed, sha256
from src.sampling_schedule import power_schedule


ROOT = Path(__file__).resolve().parent


class OriginalScheduleModel(ModelFewNFE):
    """Use the exact local non-CFG backbone with the shared four-call sampler."""

    def __init__(self):
        from src.original_meanflow.recipe import OriginalMeanFlowUNet
        Model.__init__(self, OriginalMeanFlowUNet())


def load_sweep_model(path):
    """Identify the training recipe from metadata, then load its weights strictly."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise ValueError("Expected an exported EMA checkpoint containing state_dict")
    if checkpoint.get("objective") == "mf_cfg":
        kind = "cfg"
        model = ModelFewNFE(device="cpu")
    elif checkpoint.get("recipe") in {"original_cifar_v0", "original_cifar_v0_fastgen"}:
        if checkpoint.get("nfe") != 4:
            raise ValueError("Expected the original four-NFE EMA export")
        kind = "original"
        model = OriginalScheduleModel()
    else:
        raise ValueError("Checkpoint must identify the original MeanFlow or mf_cfg recipe")
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    metadata = {key: checkpoint.get(key) for key in
                ("step", "updates", "recipe", "objective", "guidance_scale", "label_dropout")}
    metadata["model_kind"] = kind
    return model.eval(), metadata


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path,
                        default=ROOT.parent / "data/pokemon-generation-one-22k")
    parser.add_argument("--rhos", type=float, nargs="+", default=[1.0, 0.7, 1.4, 2.0])
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--reference", choices=["official_validation", "training"],
                        default="official_validation",
                        help="FID reference (default: all 3,020 official validation images).")
    parser.add_argument("--reference-seed", type=int, default=20260923,
                        help="Seed for selecting the reference in training mode only.")
    parser.add_argument("--samples-per-class", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--fid-batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-memory-fraction", type=float, default=1.0,
                        help="Cap this process's PyTorch allocator when sharing a GPU.")
    args = parser.parse_args()
    for rho in args.rhos:
        try:
            power_schedule(4, rho)
        except ValueError as error:
            parser.error(str(error))
    if len(set(args.rhos)) != len(args.rhos):
        parser.error("Each rho must be unique")
    if min(args.samples_per_class, args.batch_size, args.fid_batch_size, args.cpu_threads) < 1:
        parser.error("Sample counts, batch sizes, and CPU threads must be positive")
    if args.num_workers < 0 or not 0 < args.gpu_memory_fraction <= 1:
        parser.error("Invalid worker count or GPU memory fraction")
    return args


def write_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def emit(event, **values):
    print(json.dumps({"event": event, **values}), flush=True)


def make_reference(args):
    split_dir = ROOT / "data/pokemon-generation-one-22k"
    if not (args.data_root / ".transparency_composited_on_white").is_file():
        raise FileNotFoundError("Use an existing dataset prepared by download_dataset.py")
    mapping = json.loads((split_dir / "category_to_id.json").read_text())
    if args.reference == "official_validation":
        dataset = PokemonDataset(args.data_root, split_dir / "val_split.txt",
                                 return_category=True, category_to_id=mapping)
        if len(dataset) != 3020:
            raise ValueError("Expected all 3,020 official validation images")
        selected = dataset
        reference_paths = dataset.paths
    else:
        dataset = PokemonDataset(args.data_root, split_dir / "train_split.txt",
                                 return_category=True, category_to_id=mapping)
        # Training mode checks path separation without decoding validation images.
        validation_paths = set(read_split(split_dir / "val_split.txt"))
        if not set(dataset.paths).isdisjoint(validation_paths):
            raise ValueError("Training and validation manifests overlap")
        labels = np.asarray(dataset.category_ids)
        rng = np.random.default_rng(args.reference_seed)
        indices = []
        for category in range(151):
            candidates = np.flatnonzero(labels == category)
            if len(candidates) < args.samples_per_class:
                raise ValueError(f"Class {category} has too few training images")
            indices.extend(sorted(rng.choice(candidates, args.samples_per_class, replace=False).tolist()))
        selected = Subset(dataset, indices)
        reference_paths = [dataset.paths[index] for index in indices]
    manifest = args.output / "reference_manifest.txt"
    manifest.write_text("\n".join(reference_paths) + "\n")
    loader = DataLoader(selected, batch_size=args.batch_size,
                        shuffle=False, num_workers=args.num_workers)
    reference = args.output / f"{args.reference}_reference_png"
    prepare_reference_set(SimpleNamespace(val_dataloader=lambda: loader), reference)
    if len(list(reference.glob("*.png"))) != len(selected):
        raise RuntimeError(f"Incomplete {args.reference} reference export")
    return reference, sha256(manifest), len(selected)


def main():
    args = parse_args()
    assert_protected(ROOT)
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Missing MeanFlow checkpoint: {args.checkpoint}")
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Use a new, empty output directory for each sweep")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; run this command in the GPU host environment")
    torch.set_num_threads(args.cpu_threads)
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction, device)
    model, metadata = load_sweep_model(args.checkpoint)
    model.to(device).eval()
    if model.num_parameters >= 100_000_000:
        raise ValueError("Model exceeds the parameter budget")
    args.output.mkdir(parents=True, exist_ok=True)
    config = {key: str(value) if isinstance(value, Path) else value
              for key, value in vars(args).items()}
    official_validation = args.reference == "official_validation"
    reference_metadata = dict(reference_kind=args.reference, held_out=official_validation,
                              official_validation_used=official_validation)
    metric = ("COURSE-VALIDATION FID; local evaluation, not an official TA measurement"
              if official_validation else
              "TRAINING-reference FID; reference images may have been seen during training")
    config.update(checkpoint_sha256=sha256(args.checkpoint), checkpoint_metadata=metadata,
                  **reference_metadata, metric=metric,
                  torch_version=str(torch.__version__), gpu=torch.cuda.get_device_name(device),
                  nfe=4, parameter_count=model.num_parameters,
                  precision="FP32; TF32 disabled", status="running",
                  source_sha256={name: sha256(ROOT / name) for name in
                                 ("model.py", "sweep_sampling_schedule.py", "src/sampling_schedule.py",
                                  "evaluate.py", "dataset.py")})
    write_json(args.output / "config.json", config)
    emit("START", **config)
    started = time.monotonic()
    try:
        with isolated_seed(args.seed), evaluator_precision():
            reference, reference_hash, reference_count = make_reference(args)
            config["reference_manifest_sha256"] = reference_hash
            config["reference_count"] = reference_count
            write_json(args.output / "config.json", config)
            categories = torch.arange(151, dtype=torch.long).repeat_interleave(args.samples_per_class)
            shape = (args.batch_size, 3, 64, 64)
            warmup_categories = torch.arange(args.batch_size, device=device) % 151
            with isolated_seed(args.seed):
                for _ in range(2):
                    model.sample(shape, device=device, category=warmup_categories)
                torch.cuda.synchronize(device)
                durations = []
                for _ in range(3):
                    tick = time.monotonic()
                    model.sample(shape, device=device, category=warmup_categories)
                    torch.cuda.synchronize(device)
                    durations.append(time.monotonic() - tick)
            per_batch = statistics.median(durations)
            generation_seconds = per_batch * math.ceil(len(categories) / args.batch_size) * len(args.rhos)
            emit("GENERATION_ESTIMATE", seconds_per_batch=per_batch,
                 all_schedules_generation_minutes=generation_seconds / 60,
                 note="Excludes PNG export and FID; full ETA follows first schedule")
            rows = []
            csv_path = args.output / "fid.csv"
            for index, rho in enumerate(args.rhos):
                model.sampling_rho = rho
                run = args.output / f"rho_{rho!r}"
                tick = time.monotonic()
                torch.cuda.reset_peak_memory_stats(device)
                calls = 0

                def count_call(module, inputs, output):
                    nonlocal calls
                    calls += 1

                hook = model.backbone.register_forward_hook(count_call)
                try:
                    # Identical categories, batch partitioning, and initial noise for each rho.
                    with isolated_seed(args.seed):
                        generated = generate_samples(model, run / "generated", categories,
                                                     batch_size=args.batch_size, device=device)
                finally:
                    hook.remove()
                expected_calls = 4 * math.ceil(len(categories) / args.batch_size)
                if calls != expected_calls:
                    raise RuntimeError(f"Expected {expected_calls} backbone calls, observed {calls}")
                torch.cuda.synchronize(device)
                generation_elapsed = time.monotonic() - tick
                score = compute_fid(generated, reference, device=device, batch_size=args.fid_batch_size)
                if not math.isfinite(score):
                    raise RuntimeError("Non-finite FID")
                row = dict(rho=rho, fid=score, nfe=4, seed=args.seed,
                           generated_count=len(categories), reference_count=reference_count,
                           **reference_metadata,
                           generation_seconds=round(generation_elapsed, 2),
                           seconds=round(time.monotonic() - tick, 2),
                           backbone_calls=calls,
                           peak_gpu_gb=round(torch.cuda.max_memory_allocated(device) / 1e9, 3))
                write_json(run / "report.json", {**row, "times": power_schedule(4, rho)})
                with csv_path.open("a", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(row))
                    if not rows:
                        writer.writeheader()
                    writer.writerow(row)
                rows.append(row)
                remaining = statistics.mean(result["seconds"] for result in rows) * (len(args.rhos) - index - 1)
                emit("RESULT", **row, estimated_remaining_minutes=remaining / 60)
            best = min(rows, key=lambda row: row["fid"])
            summary = dict(status="complete", results=rows, best_rho=best["rho"],
                           best_fid=best["fid"],
                           elapsed_seconds=time.monotonic() - started,
                           **reference_metadata, reference_count=reference_count,
                           reference_manifest_sha256=reference_hash,
                           note="Single-seed schedule comparison; confirm across seeds")
            if not official_validation:
                summary.update(best_training_reference_rho=best["rho"],
                               best_training_reference_fid=best["fid"])
            write_json(args.output / "summary.json", summary)
            emit("COMPLETE", **summary)
        config["status"] = "complete"
    except BaseException as error:
        config.update(status="failed", error=f"{type(error).__name__}: {error}")
        write_json(args.output / "config.json", config)
        raise
    write_json(args.output / "config.json", config)


if __name__ == "__main__":
    main()
