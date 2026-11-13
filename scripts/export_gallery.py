"""Export real, discrete four-call trajectories for the static project page.

Run from the repository root. No training files or checkpoint weights are changed.
Only RGB pixels and explicitly allowlisted provenance leave this script.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from best_sampler import BEST_TIMES, sample_best_four
from model import ModelFewNFE

EXPECTED_SHA = "ab99f610b9bc2a8ec26f627573608b52d490065773ede28ab18bf5a940d11b41"
SPECIES = ["Pikachu", "Bulbasaur", "Charmander", "Squirtle", "Eevee", "Gengar",
           "Jigglypuff", "Psyduck", "Snorlax", "Dragonite", "Mew", "Vulpix"]
SEEDS = [42, 1234, 2026]


@torch.inference_mode()
def export(args):
    with args.checkpoint.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != EXPECTED_SHA:
        raise ValueError("Checkpoint does not match the published 101k EMA weights")
    if args.output.exists():
        raise FileExistsError("Use a fresh output directory")
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("objective") != "mf_cfg":
        raise ValueError("Expected the original MeanFlow CFG objective")
    model = ModelFewNFE(device="cpu")
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model = model.to(args.device).eval()
    mapping = json.loads((ROOT / "code/data/pokemon-generation-one-22k/category_to_id.json").read_text())
    args.output.mkdir(parents=True)
    records = []
    verification = None
    for name in SPECIES:
        for seed in SEEDS:
            torch.manual_seed(seed)
            category = torch.tensor([mapping[name]], device=args.device)
            x = torch.randn((1, 3, 64, 64), device=args.device)
            states = [x.detach().cpu()]
            for start, end in zip(BEST_TIMES[:-1], BEST_TIMES[1:]):
                t = x.new_full((1,), start)
                r = x.new_full((1,), end)
                x = x - (start - end) * model(x, t, r=r, category=category)
                if not torch.isfinite(x).all():
                    raise ValueError("Non-finite generated state")
                states.append(x.detach().cpu())
            # One narrow check: the exported trajectory ends at the existing sampler's result.
            if verification is None:
                torch.manual_seed(seed)
                expected = sample_best_four(model, (1, 3, 64, 64), device=args.device, category=category)
                torch.testing.assert_close(x.clamp(-1, 1), expected, rtol=0, atol=0)
                verification = "First trajectory exactly equals sample_best_four for the same seed"
            frames = [Image.fromarray(((s[0].clamp(-1, 1).permute(1, 2, 0).numpy() + 1) * 127.5).astype(np.uint8)) for s in states]
            slug = f"{name.lower()}-{seed}"
            strip = Image.new("RGB", (320, 64))
            for i, frame in enumerate(frames):
                strip.paste(frame, (i * 64, 0))
            strip.save(args.output / f"{slug}-steps.png", optimize=True)
            frames[-1].save(args.output / f"{slug}.png", optimize=True)
            records.append({"species": name, "class_id": mapping[name], "seed": seed,
                            "image": f"{slug}.png", "trajectory": f"{slug}-steps.png"})
            print(f"Exported {name}, seed {seed}", flush=True)
    record = {"checkpoint_sha256": digest, "checkpoint_step": 101000,
              "times": list(BEST_TIMES), "calls": 4, "resolution": [64, 64],
              "device": args.device, "torch": torch.__version__, "precision": "FP32",
              "selection": "12 named species, all three declared seeds; no sample filtering",
              "display": "Actual states clipped to [-1,1] for RGB display. No interpolated model states.",
              "verification": verification, "samples": records}
    (args.output / "gallery.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/assets/gallery")
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    export(parser.parse_args())
