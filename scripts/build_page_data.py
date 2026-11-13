"""Build browser data from an explicit allowlist, never raw server logs."""
import csv
import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs/assets"
gallery = json.loads((ASSETS / "gallery/gallery.json").read_text())
schedule = json.loads((ROOT / "data/best_schedule.json").read_text())
study_path = ASSETS / "study/study.json"
study = json.loads(study_path.read_text()) if study_path.exists() else None
history_file = ASSETS / "study/fid_history.csv" if study else ROOT / "data/fid_history.csv"
history = [[int(r["step"]), float(r["fid"])] for r in csv.DictReader(history_file.open())]
data = {
    "gallery": gallery,
    "history": sorted(history),
    "pairs": [{"seed": r["seed"], "baseline": r["baseline_fid"], "tuned": r["candidate_fid"]}
              for r in schedule["confirmation"]["paired_rows"]],
    "study": study,
}
if study:
    bins = {}
    for row in csv.DictReader((ASSETS / "study/training_loss.csv").open()):
        step = int(row["step"])
        bins.setdefault((step - 1) // 1000, []).append((step, float(row["loss"])))
    data["loss"] = [[max(r[0] for r in values), mean(r[1] for r in values),
                     min(r[1] for r in values), max(r[1] for r in values), len(values)]
                    for _, values in sorted(bins.items())]
payload = json.dumps(data, separators=(",", ":"), ensure_ascii=True)
assert "</script" not in payload.lower()
(ASSETS / "project-data.js").write_text("window.PROJECT_DATA = " + payload + ";\n")
print(f"Built {len(history)} FID observations; {len(gallery['samples'])} trajectories")
