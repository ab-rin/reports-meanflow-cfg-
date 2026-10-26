"""Generate PNGs with the selected four-call sampler or recorded baseline schedules."""

import argparse
import json
from pathlib import Path

import torch
from torchvision.transforms.functional import to_pil_image

from best_sampler import BEST_TIMES, sample_best_four
from model import ModelFewNFE, ModelOneNFE
from src.runtime import sha256
from src.sampling_schedule import power_schedule


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Fresh directory for PNGs and sampling.json")
    parser.add_argument("--schedule", choices=("best4", "uniform4", "rho2-4", "one"), default="best4")
    parser.add_argument("--classes", type=int, nargs="+", default=list(range(151)))
    parser.add_argument("--samples-per-class", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()
    if min(args.samples_per_class, args.batch_size, args.cpu_threads) < 1:
        parser.error("Sample counts, batch size and CPU threads must be positive")
    if len(set(args.classes)) != len(args.classes) or any(c < 0 or c >= 151 for c in args.classes):
        parser.error("Classes must be unique integer IDs in [0, 150]")
    return args


def load_model(path, schedule, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise ValueError("Expected an exported EMA checkpoint with state_dict")
    if checkpoint.get("objective") != "mf_cfg":
        raise ValueError("Expected the original MeanFlow CFG checkpoint")
    model_type = ModelOneNFE if schedule == "one" else ModelFewNFE
    model = model_type(device="cpu", sampling_rho=2.0 if schedule == "rho2-4" else 1.0)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    if model.num_parameters > 100_000_000:
        raise ValueError("Model exceeds the course parameter limit")
    return model.to(device).eval()


@torch.no_grad()
def main(args):
    if args.output.exists():
        raise FileExistsError("Use a fresh output directory to avoid mixing generated sets")
    torch.set_num_threads(args.cpu_threads)
    device = torch.device(args.device)
    model = load_model(args.checkpoint, args.schedule, device)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    categories = torch.tensor(args.classes, dtype=torch.long).repeat_interleave(args.samples_per_class)
    times = (BEST_TIMES if args.schedule == "best4" else
             power_schedule(model.sampling_steps, model.sampling_rho))
    args.output.mkdir(parents=True)
    image_counts = {}
    for start in range(0, len(categories), args.batch_size):
        batch = categories[start:start + args.batch_size].to(device)
        shape = (len(batch), 3, 64, 64)
        images = (sample_best_four(model, shape, device=device, category=batch)
                  if args.schedule == "best4" else model.sample(shape, device=device, category=batch))
        for image, category in zip(images, batch.tolist()):
            index = image_counts.get(category, 0)
            image_counts[category] = index + 1
            pixels = ((image.float().cpu() + 1) / 2).clamp(0, 1)
            to_pil_image(pixels).convert("RGB").save(args.output / f"{category:04d}_{index:02d}.png")
    record = {"checkpoint_sha256": sha256(args.checkpoint), "schedule": args.schedule,
              "times": list(times), "nfe": len(times) - 1, "seed": args.seed,
              "classes": args.classes, "samples_per_class": args.samples_per_class,
              "generated_count": len(categories), "batch_size": args.batch_size,
              "precision": "FP32; TF32 disabled", "device": str(device),
              "parameters": model.num_parameters}
    (args.output / "sampling.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record))


if __name__ == "__main__":
    main(parse_args())
