"""Check finite-sample sensitivity using features already extracted by the study."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from cleanfid import fid


def main(args):
    train = np.load(args.study / "private_analysis/training_features.npy")
    validation = np.load(args.study / "private_analysis/validation_features.npy")
    paths = (args.data_root / "train_split.txt").read_text().splitlines()
    groups = {}
    for index, path in enumerate(paths):
        groups.setdefault(Path(path).parent.name, []).append(index)
    assert len(groups) == 151 and len(train) == 17079 and len(validation) == 3020
    mu, covariance = validation.mean(axis=0), np.cov(validation, rowvar=False)
    results = []
    for seed in (42, 1234, 2026):
        rng = np.random.default_rng(seed)
        selected = np.concatenate([rng.choice(groups[name], 20, replace=False) for name in sorted(groups)])
        values = train[selected]
        score = float(fid.frechet_distance(values.mean(axis=0), np.cov(values, rowvar=False), mu, covariance))
        results.append({"seed": seed, "train_count": 3020, "validation_count": 3020, "fid": score,
                        "selected_indices_sha256": hashlib.sha256(selected.astype("<i8").tobytes()).hexdigest()})
        print(seed, score, flush=True)
    record = {"protocol": "20 real training images per class, without replacement, against all 3020 validation images",
              "selection": "NumPy default_rng(seed), classes sorted alphabetically; indices refer to train_split.txt",
              "features": "Reuses the exact clean-fid features from checkpoint_study.py; no network access",
              "interpretation": "Sample-count and class-balance sensitivity check, not a reconstruction of the course's unpublished baseline protocol",
              "runs": results, "mean": float(np.mean([row["fid"] for row in results])),
              "sample_sd": float(np.std([row["fid"] for row in results], ddof=1))}
    destination = args.study / "publication/baseline-sensitivity.json"
    if destination.exists():
        raise FileExistsError("Refusing to overwrite baseline results")
    destination.write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    main(parser.parse_args())
