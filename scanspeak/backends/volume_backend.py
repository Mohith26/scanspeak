"""Browser backend: executes ScanSpeak tools on real micro-CT volumes.

It owns the numbers (segmentations, volumes, diameters, HU) and emits a list of
"view ops" that the NiiVue page in web/ applies. The 3D Slicer backend in
slicer/ implements the same tool names against Slicer's own scene instead.

Data: the public TumSeg database (Jensen et al., Sci Data 2024, CC-BY). Every
animal in it (223 mice, 452 scans) is indexed in scanspeak/data/catalog.json,
with a flag for the 55 mice held out from the tumor model's training and the
experts' consensus tumor volume for every scan.
"""

import json
import os
import re

import nibabel as nib
import numpy as np
from scipy import ndimage
from scipy.spatial import ConvexHull

TUMSEG = os.environ.get(
    "TUMSEG_DIR", os.path.join(os.path.dirname(__file__), "..", "..", "data", "TumSeg database"))
CACHE = os.environ.get("SCANSPEAK_CACHE", os.path.join(os.path.dirname(__file__), "..", "..", "cache"))
TUMOR_MODEL = os.environ.get("SCANSPEAK_TUMOR_MODEL",
                             os.path.join(os.path.dirname(__file__), "..", "..", "models", "tumor_unet.pt"))

with open(os.path.join(os.path.dirname(__file__), "..", "data", "catalog.json")) as _f:
    ANIMALS = {a["id"]: a for a in json.load(_f)["animals"]}

WINDOWS = {  # (min HU, max HU)
    "soft_tissue": (-200, 300),
    "bone": (-100, 1000),
    "lung": (-400, 100),
    "full_range": (-400, 1000),
}
COLORS = {"tumor": "red", "body": "blue", "bone": "warm", "lungs": "green"}


def _tp_label(raw):          # "0d" -> "day0", "24h" -> "24h"
    m = re.fullmatch(r"(\d+)d", raw)
    return f"day{m.group(1)}" if m else raw


def find_scans(animal_id):
    """timepoint -> absolute CT path, in time order (catalog is pre-sorted)."""
    return {s["tp"]: os.path.join(TUMSEG, s["ct"]) for s in ANIMALS[animal_id]["scans"]}


def expert_volume(animal_id, tp):
    for s in ANIMALS[animal_id]["scans"]:
        if s["tp"] == tp:
            return s["expert_tumor_mm3"]
    return None


def normalize_tp(tp, available):
    if tp is None:
        return next(iter(available))
    t = str(tp).strip().lower().replace(" ", "").replace("_", "")
    t = re.sub(r"(hours?|hrs?)$", "h", t)
    t = re.sub(r"^(\d+)(days?|d)$", r"day\1", t)
    if re.fullmatch(r"\d+", t):
        t = "day" + t
    t = {"baseline": next(iter(available)), "first": next(iter(available)),
         "last": list(available)[-1], "latest": list(available)[-1]}.get(t, t)
    return t if t in available else None


_TP_PATTERNS = [
    (re.compile(r"\bday\s*(\d+)\b", re.I), lambda m: f"day{int(m.group(1))}"),
    (re.compile(r"\b(\d+)\s*d\b", re.I), lambda m: f"day{int(m.group(1))}"),
    (re.compile(r"\b(\d+(?:\.\d+)?)\s*(?:h|hrs?|hours?)\b", re.I), lambda m: f"{m.group(1).rstrip('0').rstrip('.') if '.' in m.group(1) else m.group(1)}h"),
]


def timepoints_in_text(text):
    """Timepoints the user literally wrote, in order: 'day 8' -> day8, '22.5 hours' -> 22.5h."""
    hits = []
    for rx, fmt in _TP_PATTERNS:
        hits += [(m.start(), fmt(m)) for m in rx.finditer(text or "")]
    seen, out = set(), []
    for _, tp in sorted(hits):
        if tp not in seen:
            seen.add(tp)
            out.append(tp)
    return out


def animals_in_text(text):
    """Animal ids the user literally wrote: 'D8-M37', 'dataset 4 mouse 10'."""
    t = text or ""
    found = [f"D{int(a)}-M{int(b):02d}" for a, b in re.findall(r"\bD\s*(\d+)\s*[-_ ]?\s*M\s*(\d+)\b", t, re.I)]
    found += [f"D{int(a)}-M{int(b):02d}" for a, b in re.findall(r"dataset\s*(\d+)\D{0,12}?mouse\s*M?\s*(\d+)", t, re.I)]
    return [f for i, f in enumerate(found) if f in ANIMALS and f not in found[:i]]


def snap_tp(tp, available):
    """If tp isn't a real scan but is within an hour of exactly one, return that one."""
    def hrs(x):
        m = re.fullmatch(r"day(\d+)", x)
        if m:
            return int(m.group(1)) * 24.0
        m = re.fullmatch(r"([\d.]+)h", x)
        return float(m.group(1)) if m else None
    h = hrs(tp or "")
    if h is None:
        return None
    near = [a for a in available if hrs(a) is not None and abs(hrs(a) - h) <= 1.0]
    return near[0] if len(near) == 1 else None


def resolve_animal(m):
    """Accept 'D8-M37', 'd8 m37', 'dataset 8 mouse 37', '8-37', or a bare 'M37' if unique.

    Returns (animal_id, None) or (None, error message listing the candidates).
    """
    t = str(m).strip().upper()
    nums = [int(x) for x in re.findall(r"\d+", t)]
    if len(nums) >= 2:
        ds, mn = nums[0], nums[1]
        cand = [a for a in ANIMALS.values() if a["dataset"] == ds and int(a["mouse"][1:]) == mn]
    elif len(nums) == 1:
        cand = [a for a in ANIMALS.values() if int(a["mouse"][1:]) == nums[0]]
    else:
        cand = []
    if len(cand) == 1:
        return cand[0]["id"], None
    if not cand:
        return None, f"No animal matches {m!r}. IDs look like D8-M37; open the animal list to browse all {len(ANIMALS)}."
    ids = ", ".join(a["id"] + (" (held out)" if a["held_out"] else "") for a in cand)
    return None, f"{m} exists in {len(cand)} datasets: {ids}. Say which one, e.g. 'open {cand[0]['id']}'."


class TumorModel:
    """Wraps a Fauxgraft-trained 3D U-Net (runs at 0.42 mm, output upsampled)."""

    def __init__(self, path):
        import torch
        from monai.networks.nets import UNet
        self.torch = torch
        self.net = UNet(3, 1, 1, channels=(16, 32, 64, 128), strides=(2, 2, 2), num_res_units=1)
        self.net.load_state_dict(torch.load(path, map_location="cpu"))
        self.net.eval()

    def __call__(self, ct):
        from monai.inferers import sliding_window_inference
        torch = self.torch
        s = [d - d % 2 for d in ct.shape]
        small = ct[: s[0], : s[1], : s[2]].reshape(s[0] // 2, 2, s[1] // 2, 2, s[2] // 2, 2).mean((1, 3, 5))
        x = torch.from_numpy(((small + 400.0) / 700.0 - 1.0).astype(np.float32))[None, None]
        with torch.no_grad():
            prob = torch.sigmoid(sliding_window_inference(x, (64, 64, 64), 4, self.net,
                                                          overlap=0.25, mode="gaussian"))
            prob = torch.nn.functional.interpolate(prob, scale_factor=2, mode="trilinear")[0, 0].numpy()
        full = np.zeros(ct.shape, bool)
        full[: s[0], : s[1], : s[2]] = prob > 0.5
        return full


class VolumeBackend:
    def __init__(self):
        os.makedirs(CACHE, exist_ok=True)
        self.tumor_model = TumorModel(TUMOR_MODEL) if TUMOR_MODEL and os.path.exists(TUMOR_MODEL) else None
        self.scan = None          # (mouse, timepoint)
        self.ct = None
        self.affine = None
        self.zooms = None
        self.masks = {}
        self.window = "soft_tissue"
        self.layout = "four_up"
        self.measurements = []
        self._mask_cache = {}
        self.utterance = ""   # the user's literal words for the current request

    # ---------------------------------------------------------- helpers
    def state_text(self):
        n = len(ANIMALS)
        if self.scan is None:
            return (f"no scan is open; {n} mice are available, with ids like D8-M37 "
                    "(dataset 8, mouse M37); pass the id exactly as the user wrote it")
        tps = ", ".join(find_scans(self.scan[0]))
        seg = ", ".join(self.masks) or "none"
        return (f"scan {self.scan[0]} {self.scan[1]} is open; timepoints for {self.scan[0]}: {tps}; "
                f"segmented: {seg}; layout: {self.layout}; window: {self.window}; "
                f"{n} mice available, ids like D8-M37; pass other ids exactly as the user wrote them")

    def _require_scan(self):
        if self.scan is None:
            raise ToolError("No scan is open yet. Try 'open M37'.")

    def _url(self, name):
        return "/cache/" + name

    def _compute_mask(self, structure, ct):
        if structure == "tumor":
            if self.tumor_model is None:
                raise ToolError("No tumor model configured (set SCANSPEAK_TUMOR_MODEL).")
            m = self.tumor_model(ct)
        else:
            body = ct > -200
            body = ndimage.binary_opening(body, iterations=2)
            lab, n = ndimage.label(body)
            if n > 1:
                body = lab == 1 + int(np.argmax(ndimage.sum(body, lab, range(1, n + 1))))
            body = ndimage.binary_fill_holes(body)
            if structure == "body":
                m = body
            elif structure == "bone":
                m = (ct > 350) & ndimage.binary_dilation(body, iterations=2)
                m = ndimage.binary_opening(m, iterations=1)
            elif structure == "lungs":
                inner = ndimage.binary_erosion(body, iterations=4)
                m = inner & (ct < -150)
                m = ndimage.binary_opening(m, iterations=2)
                lab, n = ndimage.label(m)
                if n:
                    sizes = ndimage.sum(m, lab, range(1, n + 1))
                    keep = np.argsort(sizes)[::-1][:2] + 1          # two lungs
                    m = np.isin(lab, keep[sizes[keep - 1] > 200])
            else:
                raise ToolError(f"unknown structure {structure}")
        # keep only components bigger than 2 mm^3 (removes speckle)
        lab, n = ndimage.label(m)
        if n:
            vox = float(np.prod(self.zooms))
            sizes = ndimage.sum(m, lab, range(1, n + 1)) * vox
            m = np.isin(lab, np.nonzero(sizes > 2.0)[0] + 1)
        return m

    def _mask_for(self, structure, scan=None):
        scan = scan or self.scan
        key = (scan, structure)
        if key not in self._mask_cache:
            ct = self.ct if scan == self.scan else self._load_ct(scan)[0]
            self._mask_cache[key] = self._compute_mask(structure, ct)
        return self._mask_cache[key]

    def _load_ct(self, scan):
        path = find_scans(scan[0])[scan[1]]
        img = nib.load(path)
        self._last_header = img.header.copy()
        # TumSeg stores an identity sform next to a correct 0.21 mm qform, so
        # physical units always come from pixdim (header zooms), never the sform.
        return img.get_fdata(dtype=np.float32), img.affine, img.header.get_zooms()[:3], path

    def _metric(self, mask, metric, ct):
        vox = float(np.prod(self.zooms))
        if not mask.any():
            return 0.0
        if metric == "volume":
            return float(mask.sum() * vox)
        if metric == "mean_intensity":
            return float(ct[mask].mean())
        if metric == "max_diameter":
            # largest component, Feret diameter from convex-hull vertices
            lab, n = ndimage.label(mask)
            big = lab == 1 + int(np.argmax(ndimage.sum(mask, lab, range(1, n + 1))))
            surf = big & ~ndimage.binary_erosion(big)
            pts = np.argwhere(surf) * np.array(self.zooms)
            if len(pts) > 4:
                pts = pts[ConvexHull(pts).vertices]
            d = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1))
            return float(d.max())
        raise ToolError(f"unknown metric {metric}")

    def _centroid_frac(self, mask):
        """Centroid as NiiVue texture fraction ((voxel + 0.5) / dim) per axis."""
        c = np.array(np.nonzero(mask)).mean(1)
        return ((c + 0.5) / np.array(mask.shape)).tolist()

    def _write_mask(self, structure):
        name = f"{self.scan[0]}_{self.scan[1]}_{structure}.nii.gz"
        path = os.path.join(CACHE, name)
        if not os.path.exists(path):
            hdr = self.header.copy()
            hdr.set_data_dtype(np.uint8)
            nib.save(nib.Nifti1Image(self.masks[structure].astype(np.uint8), None, header=hdr), path)
        return self._url(name)

    # ---------------------------------------------------------- tools
    def load_scan(self, mouse_id, timepoint=None):
        notes = []
        said = animals_in_text(self.utterance)
        mouse, err = resolve_animal(mouse_id)
        if len(said) == 1 and said[0] != mouse:   # the user's literal id beats the model's paraphrase
            mouse, err = said[0], None
            notes.append(f"used {mouse} from your message")
        if err:
            raise ToolError(err)
        scans = find_scans(mouse)
        tp = normalize_tp(timepoint, scans)
        said_tp = [t for t in timepoints_in_text(self.utterance) if t in scans]
        if len(said_tp) == 1 and said_tp[0] != tp:
            tp = said_tp[0]
            notes.append(f"used {tp} from your message")
        if tp is None:
            near = snap_tp(normalize_tp(timepoint, {timepoint: 1}) or str(timepoint).lower(), scans)
            if near:
                tp = near
                notes.append(f"no {timepoint} scan, using the closest: {near}")
            else:
                raise ToolError(f"{mouse} has no {timepoint} scan. Available: {', '.join(scans)}.")
        self.scan = (mouse, tp)
        self.ct, self.affine, self.zooms, path = self._load_ct(self.scan)
        self.header = self._last_header
        self.masks = {}
        name = f"{mouse}_{tp}_ct.nii.gz"
        dst = os.path.join(CACHE, name)
        if not os.path.lexists(dst):
            os.symlink(path, dst)
        lo, hi = WINDOWS[self.window]
        a = ANIMALS[mouse]
        note = "held out from the tumor model's training" if a["held_out"] else \
            "note: the tumor model trained on this mouse, so its tumor numbers here are optimistic"
        if notes:
            note = "; ".join(notes) + "; " + note
        return {"text": f"Opened {mouse} at {tp} ({note}).",
                "ops": [{"op": "load", "url": self._url(name), "min": lo, "max": hi}]}

    def set_window(self, preset):
        self.window = preset
        lo, hi = WINDOWS[preset]
        return {"text": f"Window set to {preset.replace('_', ' ')} ({lo} to {hi} HU).",
                "ops": [{"op": "window", "min": lo, "max": hi}]}

    def segment(self, structure):
        self._require_scan()
        self.masks[structure] = self._mask_for(structure)
        url = self._write_mask(structure)
        vol = self._metric(self.masks[structure], "volume", self.ct)
        how = "AI model" if structure == "tumor" else "intensity rules"
        return {"text": f"Segmented {structure} ({how}): {vol:,.1f} mm³.",
                "ops": [{"op": "overlay", "structure": structure, "url": url,
                         "color": COLORS[structure], "opacity": 0.5}]}

    def measure(self, structure, metric):
        self._require_scan()
        ops = []
        if structure not in self.masks:
            ops += self.segment(structure)["ops"]
        val = self._metric(self.masks[structure], metric, self.ct)
        unit = {"volume": "mm³", "max_diameter": "mm", "mean_intensity": "HU"}[metric]
        self.measurements.append({"mouse": self.scan[0], "timepoint": self.scan[1],
                                  "structure": structure, "metric": metric,
                                  "value": round(val, 2), "unit": unit})
        extra = ""
        if structure == "tumor" and metric == "volume":
            ev = expert_volume(*self.scan)
            if ev is not None:
                extra = f" Experts' consensus outline: {ev:,.1f} mm³."
        return {"text": f"{structure} {metric.replace('_', ' ')} at {self.scan[1]}: {val:,.1f} {unit}.{extra}",
                "ops": ops, "value": val}

    def compare_timepoints(self, structure, metric, from_timepoint, to_timepoint):
        self._require_scan()
        scans = find_scans(self.scan[0])
        a, b = normalize_tp(from_timepoint, scans), normalize_tp(to_timepoint, scans)
        said_tp = [t for t in timepoints_in_text(self.utterance) if t in scans]
        if len(said_tp) == 2 and [a, b] != said_tp:   # literal timepoints beat the model's paraphrase
            a, b = said_tp
        a = a or snap_tp(str(from_timepoint).lower(), scans)
        b = b or snap_tp(str(to_timepoint).lower(), scans)
        if a is None or b is None:
            raise ToolError(f"{self.scan[0]} has timepoints {', '.join(scans)}.")
        vals = []
        for tp in (a, b):
            sc = (self.scan[0], tp)
            ct = self.ct if sc == self.scan else self._load_ct(sc)[0]
            vals.append(self._metric(self._mask_for(structure, sc), metric, ct))
        unit = {"volume": "mm³", "max_diameter": "mm", "mean_intensity": "HU"}[metric]
        delta = vals[1] - vals[0]
        pct = delta / vals[0] * 100 if vals[0] else float("nan")
        for tp, v in zip((a, b), vals):
            self.measurements.append({"mouse": self.scan[0], "timepoint": tp, "structure": structure,
                                      "metric": metric, "value": round(v, 2), "unit": unit})
        extra = ""
        if structure == "tumor" and metric == "volume":
            e0, e1 = expert_volume(self.scan[0], a), expert_volume(self.scan[0], b)
            if e0 and e1:
                extra = f" Experts' consensus: {e0:,.1f} → {e1:,.1f} mm³ ({(e1 - e0) / e0 * 100:+.0f}%)."
        return {"text": (f"{structure} {metric.replace('_', ' ')}: {vals[0]:,.1f} {unit} at {a} → "
                         f"{vals[1]:,.1f} {unit} at {b} ({delta:+,.1f} {unit}, {pct:+.0f}%).{extra}"),
                "ops": [], "value": pct}

    def set_layout(self, layout):
        self.layout = layout
        pretty = {"three_d": "3D render", "four_up": "four-up"}.get(layout, layout)
        return {"text": f"Layout: {pretty}.", "ops": [{"op": "layout", "layout": layout}]}

    def focus_on(self, structure):
        self._require_scan()
        ops = []
        if structure not in self.masks:
            ops += self.segment(structure)["ops"]
        if not self.masks[structure].any():
            return {"text": f"No {structure} found to focus on.", "ops": ops}
        return {"text": f"Centered on the {structure}.",
                "ops": ops + [{"op": "focus", "frac": self._centroid_frac(self.masks[structure])}]}

    def set_visibility(self, structure, visible):
        self._require_scan()
        ops = []
        if visible and structure not in self.masks:
            ops += self.segment(structure)["ops"]
        return {"text": f"{structure} overlay {'shown' if visible else 'hidden'}.",
                "ops": ops + [{"op": "opacity", "structure": structure, "opacity": 0.5 if visible else 0.0}]}

    def set_opacity(self, structure, opacity):
        self._require_scan()
        opacity = float(min(max(opacity, 0.0), 1.0))
        ops = []
        if structure not in self.masks:
            ops += self.segment(structure)["ops"]
        return {"text": f"{structure} opacity {opacity:.0%}.",
                "ops": ops + [{"op": "opacity", "structure": structure, "opacity": opacity}]}

    def export_measurements(self, format):
        name = f"measurements.{format}"
        path = os.path.join(CACHE, name)
        if format == "json":
            with open(path, "w") as f:
                json.dump(self.measurements, f, indent=1)
        else:
            with open(path, "w") as f:
                f.write("mouse,timepoint,structure,metric,value,unit\n")
                for m in self.measurements:
                    f.write(",".join(str(m[k]) for k in
                                     ["mouse", "timepoint", "structure", "metric", "value", "unit"]) + "\n")
        return {"text": f"Saved {len(self.measurements)} measurements.",
                "ops": [{"op": "download", "url": self._url(name)}]}

    def reset_view(self):
        self.window, self.layout = "soft_tissue", "four_up"
        lo, hi = WINDOWS[self.window]
        return {"text": "View reset.", "ops": [{"op": "reset", "min": lo, "max": hi}]}

    # ---------------------------------------------------------- dispatch
    def execute(self, calls, utterance=""):
        self.utterance = utterance
        results = []
        for i, c in enumerate(calls):
            if results and not results[-1]["ok"]:   # never run later steps on the wrong scan
                results.append({"call": c, "ok": False, "ops": [],
                                "text": f"(skipped {c['name']} because the step before it failed)"})
                continue
            fn = getattr(self, c["name"], None)
            try:
                if fn is None or c["name"].startswith("_"):
                    raise ToolError(f"unknown tool {c['name']}")
                r = fn(**(c.get("arguments") or {}))
                results.append({"call": c, "ok": True, **r})
            except ToolError as e:
                results.append({"call": c, "ok": False, "text": str(e), "ops": []})
            except TypeError as e:
                results.append({"call": c, "ok": False, "text": f"bad arguments: {e}", "ops": []})
        return results


class ToolError(Exception):
    pass
