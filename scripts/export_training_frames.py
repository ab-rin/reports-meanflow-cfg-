"""Export eight fixed Charmander seeds from every evaluated EMA checkpoint.

Uses existing weights and model source. Never downloads or writes into the run.
The output must be a fresh directory and contains images plus public provenance.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import torch
from torchvision.transforms.functional import to_pil_image

MODEL_SHA256 = "6eb5125a45a11d6c3e5a676717e786f3351533575c17f7505c46524b1b631fca"
SEEDS = [7, 19, 42, 73, 101, 256, 512, 777]


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@torch.inference_mode()
def main(args):
    if args.output.exists():
        raise FileExistsError("Use a fresh output directory")
    if sha(args.source_root / "model.py") != MODEL_SHA256:
        raise ValueError("Model source differs from the published implementation")
    sys.path.insert(0, str(args.source_root))
    from model import ModelFewNFE

    rows = list(csv.DictReader((args.run_root / "fid.csv").open()))
    steps = sorted({int(row["step"]) for row in rows if int(row["nfe"]) == 4})
    paths = [args.run_root / "checkpoints" / f"ema_{step:07d}.ckpt" for step in steps]
    if not steps or not all(path.is_file() for path in paths):
        raise ValueError("Every evaluated step must have its own retained EMA checkpoint")
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model = ModelFewNFE(device="cpu").to(args.device).eval()
    category = torch.tensor([14], device=args.device)
    record = {
        "species": "Charmander", "class_id": 14, "seed": 42, "seeds": SEEDS,
        "times": [1., .75, .5, .25, 0.], "nfe": 4,
        "precision": "FP32; TF32 disabled", "device": args.device,
        "torch": torch.__version__, "model_sha256": MODEL_SHA256,
        "selection": "Every retained EMA checkpoint with a recorded uniform-four FID; no image filtering",
        "frames": [], "complete": False,
    }
    args.output.mkdir(parents=True)
    for step, path in zip(steps, paths):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint.get("objective") != "mf_cfg" or int(checkpoint["step"]) != step:
            raise ValueError(f"Checkpoint metadata mismatch at {step}")
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        samples = []
        for seed in SEEDS:
            torch.manual_seed(seed)
            pixels = model.sample((1, 3, 64, 64), device=args.device, category=category)[0]
            if not torch.isfinite(pixels).all():
                raise ValueError(f"Non-finite image at {step}, seed {seed}")
            image = to_pil_image(((pixels.float().cpu() + 1) / 2).clamp(0, 1)).convert("RGB")
            name = f"journey-charmander-{step}.png" if seed == 42 else f"journey-charmander-{seed}-{step}.png"
            image.save(args.output / name)
            samples.append({"seed": seed, "image": name, "image_sha256": sha(args.output / name)})
        primary = next(sample for sample in samples if sample["seed"] == 42)
        record["frames"].append({"step": step, "image": primary["image"], "image_sha256": primary["image_sha256"],
                                  "samples": samples, "checkpoint_sha256": sha(path)})
        (args.output / "training-frames.json").write_text(json.dumps(record, indent=2) + "\n")
        print(f"Exported {step} ({len(record['frames'])}/{len(steps)})", flush=True)
        del checkpoint
    record["complete"] = True
    (args.output / "training-frames.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    main(parser.parse_args())
