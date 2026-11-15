"""Create display-only alpha masks while retaining every source RGB byte.

Uses a predownloaded, checksum-verified BiRefNet model with rembg-compatible preprocessing. Source samples,
training data and FID measurements are never modified. Initial noise stays whole.
"""
import argparse
import hashlib
import json
from pathlib import Path

from collections import deque

import numpy as np
from scipy.ndimage import binary_erosion, label
from PIL import Image, ImageDraw
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs/assets"
OUTLINE_PATH = ROOT / "data/display-mask-outlines.json"
OUTLINES = {row["rgb_sha256"]: row for row in json.loads(OUTLINE_PATH.read_text())["outlines"]} if OUTLINE_PATH.exists() else {}
MODEL_MD5 = "7a35a0141cbbc80de11d9c9a28f52697"


def file_hash(path, algorithm="sha256"):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, algorithm).hexdigest()


def make_session(model, device="cpu"):
    if file_hash(model, "md5") != MODEL_MD5:
        raise ValueError("Expected the verified, already-downloaded BiRefNet model")
    options = ort.SessionOptions()
    options.inter_op_num_threads = 1
    options.intra_op_num_threads = 4
    providers = ["CPUExecutionProvider"]
    if device == "cuda":
        import torch  # Load the existing CUDA/cuDNN libraries; no model code downloads.
        torch.cuda.init()
        providers.insert(0, ("CUDAExecutionProvider", {"gpu_mem_limit": 8 * 1024 ** 3}))
    session = ort.InferenceSession(str(model), sess_options=options, providers=providers)
    if device == "cuda" and session.get_providers()[0] != "CUDAExecutionProvider":
        raise RuntimeError("The requested GPU provider did not initialize")
    return session


def predict_mask(image, session):
    # The normalization and sigmoid follow rembg's BiRefNet session protocol.
    resized = np.asarray(image.resize((1024, 1024), Image.Resampling.LANCZOS), dtype=np.float32)
    resized /= max(float(resized.max()), 1e-6)
    normalized = (resized - np.array([.485, .456, .406], dtype=np.float32)) / np.array([.229, .224, .225], dtype=np.float32)
    tensor = normalized.transpose(2, 0, 1)[None].astype(np.float32)
    logits = session.run(None, {session.get_inputs()[0].name: tensor})[0][0, 0]
    prediction = 1 / (1 + np.exp(-np.clip(logits, -30, 30)))
    low, high = prediction.min(), prediction.max()
    prediction = (prediction - low) / max(float(high - low), 1e-8)
    mask = Image.fromarray((prediction.clip(0, 1) * 255).astype(np.uint8))
    return mask.resize(image.size, Image.Resampling.LANCZOS)


def outline_alpha(key):
    row = OUTLINES.get(key)
    if not row:
        return None
    scale = 8
    mask = Image.new("L", (64 * scale, 64 * scale))
    draw = ImageDraw.Draw(mask)
    for polygon in row["polygons"]:
        draw.polygon([(round(x * scale), round(y * scale)) for x, y in polygon], fill=255)
    for stroke in row["strokes"]:
        draw.line([(round(x * scale), round(y * scale)) for x, y in stroke["points"]], fill=255,
                  width=round(stroke["width"] * scale), joint="curve")
    for polygon in row.get("clear", []):
        draw.polygon([(round(x * scale), round(y * scale)) for x, y in polygon], fill=0)
    return mask.resize((64, 64), Image.Resampling.LANCZOS)


def cutout(image, session, cache):
    image = image.convert("RGB")
    key = hashlib.sha256(image.tobytes()).hexdigest()
    if key not in cache:
        explicit = outline_alpha(key)
        if explicit is not None:
            result = image.convert("RGBA")
            result.putalpha(explicit)
            cache[key] = result
            return result
        alpha = np.asarray(predict_mask(image, session), dtype=np.float32)
        if not np.isfinite(alpha).all():
            raise ValueError("Invalid foreground mask")
        alpha = np.clip((alpha - 10) * 255 / 235, 0, 255).astype(np.uint8)
        neural_alpha = alpha.copy()
        # Flat pale backgrounds have useful color evidence even where a neural
        # mask misses a dark forehead or a faint tail. Protect interior whites.
        rgb = np.asarray(image, dtype=np.float32)
        border = np.concatenate((rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]))
        background = np.median(border, axis=0)
        flat = np.mean(np.max(np.abs(border - background), axis=1) < 14) > .6
        if flat and background.min() > 200:
            eligible = np.max(np.abs(rgb - background), axis=2) < 14
            eligible &= ~binary_erosion(alpha > 220)
            height, width = eligible.shape
            connected = np.zeros_like(eligible)
            queue = deque([(x, y) for y in range(height) for x in range(width)
                           if x in (0, width - 1) or y in (0, height - 1)])
            while queue:
                x, y = queue.popleft()
                if not (0 <= x < width and 0 <= y < height) or connected[y, x] or not eligible[y, x]:
                    continue
                connected[y, x] = True
                queue.extend(((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)))
            alpha = np.where(connected, 0, 255).astype(np.uint8)
            components, count = label(alpha > 0, structure=np.ones((3, 3)))
            for component in range(1, count + 1):
                region = components == component
                touches_edge = region[0].any() or region[-1].any() or region[:, 0].any() or region[:, -1].any()
                if touches_edge and region.sum() < alpha.size * .1 and neural_alpha[region].mean() < 32:
                    alpha[region] = 0
        result = image.convert("RGBA")
        result.putalpha(Image.fromarray(alpha))
        assert np.array_equal(np.asarray(result)[:, :, :3], np.asarray(image))
        cache[key] = result
    return cache[key]


def main(args):
    if args.output.exists():
        raise FileExistsError("Use a fresh cutout directory")
    session = make_session(args.model, args.device)
    cache = {}
    (args.output / "gallery").mkdir(parents=True)
    (args.output / "study").mkdir()
    files = []

    def save(image, kind, name):
        destination = args.output / kind / name
        image.save(destination, optimize=True)
        files.append(f"{kind}/{name}")

    gallery = json.loads((ASSETS / "gallery/gallery.json").read_text())
    for i, sample in enumerate(gallery["samples"]):
        image = Image.open(ASSETS / "gallery" / sample["image"]).convert("RGB")
        save(cutout(image, session, cache), "gallery", sample["image"])
        strip = Image.open(ASSETS / "gallery" / sample["trajectory"]).convert("RGB")
        output = strip.convert("RGBA")
        for frame in range(1, 5):
            tile = strip.crop((frame * 64, 0, (frame + 1) * 64, 64))
            output.paste(cutout(tile, session, cache), (frame * 64, 0))
        save(output, "gallery", sample["trajectory"])
        if (i + 1) % 12 == 0: print(f"Gallery {i + 1}/{len(gallery['samples'])}", flush=True)

    study = json.loads((ASSETS / "study/study.json").read_text())
    for checkpoint in study["checkpoints"]:
        for seed in study["visual_seeds"]:
            grid = Image.new("RGBA", (256, 192))
            for i, name in enumerate(study["species"]):
                filename = f"compare-{checkpoint['step']}-{name.lower()}-{seed}.png"
                image = cutout(Image.open(ASSETS / "study" / filename), session, cache)
                save(image, "study", filename)
                grid.paste(image, ((i % 4) * 64, (i // 4) * 64))
            save(grid, "study", f"compare-{checkpoint['step']}-{seed}.png")
    frames = json.loads((ASSETS / "study/training-frames.json").read_text())["frames"]
    columns = 14
    atlas = Image.new("RGBA", (columns * 64, ((len(frames) + columns - 1) // columns) * 64))
    for i, frame in enumerate(frames):
        image = cutout(Image.open(ASSETS / "study" / frame["image"]), session, cache)
        save(image, "study", frame["image"])
        atlas.paste(image, ((i % columns) * 64, (i // columns) * 64))
    save(atlas, "study", "training-atlas.png")
    record = {
        "purpose": "Presentation-only foreground masks; not native generator output",
        "model": "BiRefNet-general; preprocessing follows rembg 2.0.67", "onnxruntime": ort.__version__,
        "model_source": "https://github.com/danielgatis/rembg/releases/download/v0.0.0/BiRefNet-general-epoch_244.onnx",
        "device": args.device, "provider": session.get_providers()[0],
        "model_md5": MODEL_MD5, "model_sha256": file_hash(args.model),
        "rgb_policy": "Every source RGB byte is retained; only alpha is added. No repainting, resizing or recoloring.",
        "alpha_policy": "BiRefNet alpha; flat pale backgrounds use edge-connected color masking with protected foreground interiors; reviewed polygon corrections for difficult backgrounds",
        "noise_policy": "Initial Gaussian noise is shown in full; foreground masks apply to the four subsequent states",
        "originals": "Original RGB images remain in assets/gallery and assets/study",
        "outline_corrections": len(OUTLINES),
        "outline_source": "data/display-mask-outlines.json",
        "outline_source_sha256": file_hash(OUTLINE_PATH),
        "files": files,
    }
    (args.output / "manifest.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"Saved {len(files)} transparent presentation assets", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--output", type=Path, default=ASSETS / "cutouts")
    main(parser.parse_args())
