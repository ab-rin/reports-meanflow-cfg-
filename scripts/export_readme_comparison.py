"""Export the README's four species with identical CPU noise across all rows."""

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

CHECKPOINTS = {
    64000: "8ba96595edd883380a489060d5aa58bd346242ae8279c6491b8fcc80d520ec14",
    101000: "ab99f610b9bc2a8ec26f627573608b52d490065773ede28ab18bf5a940d11b41",
}
SPECIES = ["Pikachu", "Bulbasaur", "Charmander", "Squirtle"]


@torch.inference_mode()
def main(args):
    if args.output.exists():
        raise FileExistsError("Use a fresh output directory")
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True)
    mapping = json.loads((ROOT / "code/data/pokemon-generation-one-22k/category_to_id.json").read_text())
    torch.manual_seed(42)
    noise = torch.randn((1, 3, 64, 64))
    noise_sha = hashlib.sha256(noise.numpy().tobytes()).hexdigest()
    records = []
    args.output.mkdir(parents=True)
    for step, expected_sha in CHECKPOINTS.items():
        checkpoint_path = args.checkpoint_dir / f"ema_{step:07d}.ckpt"
        with checkpoint_path.open("rb") as stream:
            assert hashlib.file_digest(stream, "sha256").hexdigest() == expected_sha
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        assert checkpoint["objective"] == "mf_cfg"
        model = ModelFewNFE(device="cpu").eval()
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        assert model.num_parameters == 56010243
        schedules = [("uniform4", (1., .75, .5, .25, 0.))]
        if step == 101000:
            schedules.append(("tuned4", BEST_TIMES))
        for label, times in schedules:
            for species in SPECIES:
                category = torch.tensor([mapping[species]])
                calls = []
                hook = model.register_forward_pre_hook(lambda module, inputs: calls.append(inputs[0].clone()))
                torch.manual_seed(42)
                if label == "uniform4":
                    output = model.sample((1, 3, 64, 64), device="cpu", category=category)
                else:
                    output = sample_best_four(model, (1, 3, 64, 64), device="cpu", category=category)
                hook.remove()
                assert len(calls) == 4
                torch.testing.assert_close(calls[0], noise, rtol=0, atol=0)
                assert torch.isfinite(output).all()
                pixels = ((output[0].permute(1, 2, 0).numpy() + 1) * 127.5).astype(np.uint8)
                filename = f"{step}-{label}-{species.lower()}-42.png"
                image_path = args.output / filename
                Image.fromarray(pixels).save(image_path, optimize=True)
                # The tuned row must reproduce the already published CPU gallery.
                if label == "tuned4":
                    reference = ROOT / "docs/assets/gallery" / f"{species.lower()}-42.png"
                    np.testing.assert_array_equal(pixels, np.asarray(Image.open(reference).convert("RGB")))
                records.append({"checkpoint_step": step, "checkpoint_sha256": expected_sha,
                                "schedule": label, "times": list(times), "species": species,
                                "class_id": mapping[species], "image": filename,
                                "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest()})
                print(f"Exported {filename}", flush=True)
        del model, checkpoint
    record = {"seed": 42, "initial_noise_sha256": noise_sha, "device": "cpu",
              "torch": torch.__version__, "precision": "FP32", "nfe": 4,
              "selection": "First four species from the existing study; no image filtering",
              "verification": "Identical initial noise and four calls for every image; tuned row matches published gallery pixels",
              "samples": records}
    (args.output / "comparison.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
