"""Reproduce the page's matched checkpoint study using existing local assets.

Never downloads weights or data. Writes only to a new --output directory.
The private_analysis directory is intermediate data, not a publication asset.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image

SPECIES = ["Pikachu", "Bulbasaur", "Charmander", "Squirtle", "Eevee", "Gengar",
           "Jigglypuff", "Psyduck", "Snorlax", "Dragonite", "Mew", "Vulpix"]
SEEDS = [42, 1234, 2026]
TIMES = [1., .75, .5, .25, 0.]


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def rgb(tensor):
    return to_pil_image(((tensor.float().cpu() + 1) / 2).clamp(0, 1)).convert("RGB")


@torch.inference_mode()
def main(args):
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite an existing study")
    sys.path.insert(0, str(args.source_root))
    from model import ModelFewNFE
    from cleanfid import fid
    from cleanfid.inception_torchscript import InceptionV3W
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if not Path("/tmp/inception-2015-12-05.pt").is_file():
        raise FileNotFoundError("Expected already-cached Inception weights; no download attempted")
    if sha(args.source_root / "model.py") != "6eb5125a45a11d6c3e5a676717e786f3351533575c17f7505c46524b1b631fca":
        raise ValueError("Model source differs from the repository export")
    args.output.mkdir(parents=True)
    public = args.output / "publication"
    private = args.output / "private_analysis"
    public.mkdir()
    private.mkdir()
    mapping = json.loads((args.data_root / "category_to_id.json").read_text())
    record = {
        "purpose": "Matched checkpoint comparison and real-data FID reference",
        "device": torch.cuda.get_device_name(), "torch": torch.__version__,
        "precision": "FP32; TF32 disabled", "times": TIMES, "nfe": 4,
        "sample_policy": "12 predeclared species, all 3 seeds, no image filtering",
        "comparison_selection": "64k is the first recorded uniform-four checkpoint below 22; 101k is the best recorded uniform-four checkpoint",
        "visual_seeds": SEEDS, "species": SPECIES, "evaluation_seed": 1234,
        "evaluation_batch": 32, "generated_count": 3020, "reference_count": 3020,
        "train_manifest_sha256": sha(args.data_root / "train_split.txt"),
        "val_manifest_sha256": sha(args.data_root / "val_split.txt"),
        "inception_sha256": sha("/tmp/inception-2015-12-05.pt"),
        "checkpoints": [],
    }
    def save_record():
        (public / "study.json").write_text(json.dumps(record, indent=2) + "\n")

    # Export only numeric fields; logs and server paths never enter the public record.
    history = {}
    for line in (args.run_root / "metrics.jsonl").read_text().splitlines():
        row = json.loads(line)
        history[int(row["step"])] = {key: row[key] for key in ("step", "loss", "training_seconds")}
    with (public / "training_loss.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["step", "loss", "training_seconds"])
        writer.writeheader()
        writer.writerows(history[key] for key in sorted(history))
    rows = [r for r in csv.DictReader((args.run_root / "fid.csv").open()) if int(r["nfe"]) == 4]
    with (public / "fid_history.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["step", "fid", "nfe", "seed"])
        writer.writeheader()
        writer.writerows({key: row[key] for key in writer.fieldnames} for row in rows)
    record["last_logged_step"] = max(history)
    record["retained_step"] = json.loads((args.run_root / "status.json").read_text())["step"]
    record["loss_export"] = "Last occurrence per step; only step, loss and cumulative active seconds"
    record["active_hours_at_64k"] = history[64000]["training_seconds"] / 3600
    record["active_hours_at_101k"] = history[101000]["training_seconds"] / 3600
    save_record()

    def load(step):
        path = args.run_root / "checkpoints" / f"ema_{step:07d}.ckpt"
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        model = ModelFewNFE(device="cpu")
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        return model.cuda().eval(), path

    # Small visual comparison, deliberately fixed before viewing the images.
    for step in (64000, 101000):
        model, path = load(step)
        item = {"step": step, "sha256": sha(path),
                "historical_uniform_fid": float(next(row["fid"] for row in rows if int(row["step"]) == step))}
        record["checkpoints"].append(item)
        for seed in SEEDS:
            sheet = Image.new("RGB", (4 * 64, 3 * 64), "white")
            for i, name in enumerate(SPECIES):
                torch.manual_seed(seed)
                image = model.sample((1, 3, 64, 64), category=torch.tensor([mapping[name]], device="cuda"))[0]
                picture = rgb(image)
                picture.save(public / f"compare-{step}-{name.lower()}-{seed}.png")
                sheet.paste(picture, ((i % 4) * 64, (i // 4) * 64))
            sheet.save(public / f"compare-{step}-{seed}.png")
        print(f"Exported matched visual samples at {step}", flush=True)
        del model
        torch.cuda.empty_cache()
    # Training snapshots reuse identical per-class seeds and uniform sampling.
    for step in (1000, 8000, 32000, 64000, 101000, 122000):
        model, _ = load(step)
        torch.manual_seed(42)
        image = model.sample((1, 3, 64, 64), category=torch.tensor([mapping["Pikachu"]], device="cuda"))[0]
        rgb(image).save(public / f"journey-{step}.png")
        del model
        torch.cuda.empty_cache()
    save_record()

    # Use the already-cached official feature network explicitly, with downloads disabled.
    feature_model = InceptionV3W("/tmp", download=False, resize_inside=False).cuda().eval()
    def features(directory):
        return fid.get_folder_features(str(directory), feature_model, num_workers=4,
                                       batch_size=32, device=torch.device("cuda"), mode="clean")
    def score(first, second):
        return float(fid.frechet_distance(np.mean(first, axis=0), np.cov(first, rowvar=False),
                                          np.mean(second, axis=0), np.cov(second, rowvar=False)))
    reference = args.run_root / "official_validation_reference_png"
    if len(list(reference.glob("*.png"))) != 3020:
        raise ValueError("Expected 3,020 cached course-reference PNGs")
    validation_features = features(reference)
    # Match the course Dataset resize -> normalize -> PNG conversion without invoking its downloader.
    preprocessing = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor(),
                                        transforms.Normalize((.5, .5, .5), (.5, .5, .5))])
    train_png = private / "train_png"
    train_png.mkdir()
    train_files = (args.data_root / "train_split.txt").read_text().splitlines()
    if len(train_files) != 17079:
        raise ValueError("Expected the fixed 17,079-image training split")
    for i, relative in enumerate(train_files):
        with Image.open(args.data_root / relative) as image:
            rgb(preprocessing(image.convert("RGB"))).save(train_png / f"{i:05d}.png")
    train_features = features(train_png)
    record["train_validation_fid"] = score(train_features, validation_features)
    record["train_count"] = len(train_files)
    record["reference_note"] = "All training images vs all validation images; unequal set sizes. Descriptive reference, not a lower bound."
    np.save(private / "training_features.npy", train_features)
    np.save(private / "validation_features.npy", validation_features)
    print("Train-validation FID:", record["train_validation_fid"], flush=True)
    save_record()
    for item in record["checkpoints"]:
        model, _ = load(item["step"])
        destination = private / f"evaluation-{item['step']}"
        destination.mkdir()
        categories = torch.arange(151, device="cuda").repeat_interleave(20)
        torch.manual_seed(1234)
        for start in range(0, len(categories), 32):
            batch = categories[start:start + 32]
            images = model.sample((len(batch), 3, 64, 64), category=batch)
            for offset, image in enumerate(images):
                rgb(image).save(destination / f"{start + offset:04d}.png")
        item["remeasured_uniform_fid"] = score(features(destination), validation_features)
        print("Checkpoint FID:", item["step"], item["remeasured_uniform_fid"], flush=True)
        del model
        torch.cuda.empty_cache()
        save_record()
    record["complete"] = True
    save_record()
    print("Study complete", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
