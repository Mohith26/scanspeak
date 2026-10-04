"""Index every animal in the TumSeg database into scanspeak/data/catalog.json.

    python3 scripts/build_catalog.py "/path/to/TumSeg database"

For each mouse it records its scans (timepoint, CT path, consensus path), the
experts' consensus tumor volume at each timepoint, and whether the mouse was
held out from the tumor model's training (same mouse-level split as Fauxgraft:
seed 0, 25% of mice). Paths are stored relative to the TumSeg root.
"""

import glob
import json
import os
import re
import sys

import nibabel as nib
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def fauxgraft_test_mice(all_mice):
    """Reproduce fauxgraft.train.split(): sorted mice, shuffled with seed 0, first 25% held out."""
    mice = sorted(all_mice)
    rng = np.random.default_rng(0)
    rng.shuffle(mice)
    return set(mice[: int(len(mice) * 0.25)])


def tp_label(raw):
    m = re.fullmatch(r"(\d+)d", raw)
    return f"day{m.group(1)}" if m else raw


def tp_hours(label):
    m = re.fullmatch(r"day(\d+)", label)
    if m:
        return int(m.group(1)) * 24.0
    m = re.fullmatch(r"([\d.]+)h", label)
    return float(m.group(1)) if m else 0.0


def main(src):
    by_mouse = {}
    for folder in sorted(glob.glob(os.path.join(src, "Dataset *", "*", "*"))):
        if not os.path.isdir(folder):
            continue
        ds = int(re.search(r"Dataset (\d+)", folder).group(1))
        mouse, raw_tp = os.path.basename(folder).split("_", 1)
        cts = glob.glob(os.path.join(folder, "CT_*.nii.gz"))
        stp = glob.glob(os.path.join(folder, "STAPLE_*.nii.gz"))
        if not cts:
            continue
        expert = None
        if stp:
            img = nib.load(stp[0])
            vox = float(np.prod(img.header.get_zooms()[:3]))
            expert = round(float((np.asarray(img.dataobj) > 0.5).sum() * vox), 1)
        key = (ds, mouse)
        by_mouse.setdefault(key, []).append({
            "tp": tp_label(raw_tp),
            "ct": os.path.relpath(cts[0], src),
            "expert_tumor_mm3": expert,
        })
    # held-out flag uses the Fauxgraft index's mouse keys (D08_M37); the 13 scans that
    # Fauxgraft skipped belong to mice that also have other scans, so keys still match
    keys = {f"D{ds:02d}_{m}" for ds, m in by_mouse}
    test = fauxgraft_test_mice(keys)
    animals = []
    for (ds, mouse), scans in sorted(by_mouse.items()):
        scans.sort(key=lambda s: tp_hours(s["tp"]))
        animals.append({
            "id": f"D{ds}-{mouse}",
            "dataset": ds,
            "mouse": mouse,
            "held_out": f"D{ds:02d}_{mouse}" in test,
            "scans": scans,
        })
    out = os.path.join(ROOT, "scanspeak", "data", "catalog.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump({"source": "TumSeg (Jensen et al., Sci Data 2024, CC-BY)", "animals": animals}, f, indent=1)
    n_scans = sum(len(a["scans"]) for a in animals)
    print(f"{len(animals)} animals, {n_scans} scans, {sum(a['held_out'] for a in animals)} held out -> {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "data", "TumSeg database"))
