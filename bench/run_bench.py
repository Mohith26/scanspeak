"""Run the ScanSpeak benchmark: models x decoding modes x 72 cases.

    python3 -u bench/run_bench.py                 # full grid, resumable
    python3 -u bench/run_bench.py --summarize     # just rebuild the summary

Raw results stream to results/raw.jsonl (one line per model/mode/case), so a
killed run picks up where it left off. The summary goes to results/summary.json.
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from cases import CASES, S, S_NONE, S_SEG  # noqa: E402
from heldout import HELDOUT, S2  # noqa: E402
from scanspeak.agent import normalize, plan  # noqa: E402
from scanspeak.tools import validate_call  # noqa: E402

RAW = os.path.join(ROOT, "results", "raw.jsonl")
SUMMARY = os.path.join(ROOT, "results", "summary.json")

MODELS = ["qwen2.5:1.5b", "qwen2.5:3b", "llama3.2:3b", "gemma3:4b", "llama3.1:8b"]
MODES = ["prompt", "schema", "strict", "native"]
NO_NATIVE_TOOLS = {"gemma3:4b"}  # Ollama reports no "tools" capability

INITIAL = {
    S: {"scan": ("M07", "day7"), "segmented": set(), "layout": "axial"},
    S_SEG: {"scan": ("M07", "day7"), "segmented": {"tumor", "body"}, "layout": "four_up"},
    S_NONE: {"scan": None, "segmented": set(), "layout": "axial"},
    S2: {"scan": ("M22", "day3"), "segmented": set(), "layout": "four_up"},
}
SETS = {"main": (CASES, RAW), "heldout": (HELDOUT, os.path.join(ROOT, "results", "raw_heldout.jsonl"))}


# ---------------------------------------------------------------- scoring
def norm_mouse(v):
    m = re.search(r"(\d+)", str(v))
    return f"M{int(m.group(1)):02d}" if m else str(v).upper()


def norm_tp(v):
    if v is None:
        return "default"
    v = normalize(v)
    v = v.replace("days", "day").replace("d", "day") if re.fullmatch(r"\d+d", v) else v
    if re.fullmatch(r"\d+", v):
        v = "day" + v
    if v in ("day0", "baseline", "first"):
        return "default_or_day0"
    return v


def norm_args(call):
    out = {}
    for k, v in (call.get("arguments") or {}).items():
        if k == "mouse_id":
            out[k] = norm_mouse(v)
        elif k in ("timepoint", "from_timepoint", "to_timepoint"):
            out[k] = norm_tp(v)
        elif k == "opacity" and isinstance(v, (int, float)):
            out[k] = round(float(v), 3)
        else:
            out[k] = normalize(v)
    if call.get("name") == "load_scan" and "timepoint" not in out:
        out["timepoint"] = "default_or_day0"
    return out


def exact_match(pred, gold):
    if len(pred) != len(gold):
        return False
    return all(p.get("name") == g["name"] and norm_args(p) == norm_args(g)
               for p, g in zip(pred, gold))


def simulate(calls, state_key):
    """Execute calls on a toy viewer; return a comparable final state or None if invalid."""
    init = INITIAL[state_key]
    st = {
        "scan": init["scan"],
        "segmented": set(init["segmented"]),
        "visible": {s: True for s in init["segmented"]},
        "opacity": {},
        "layout": init["layout"],
        "window": "soft_tissue",
        "focus": None,
        "reported": set(),
        "exports": set(),
    }
    for call in calls:
        if validate_call(call):
            return None
        a = norm_args(call)
        name = call["name"]
        if name == "load_scan":
            new = (a["mouse_id"], a["timepoint"])
            cur = st["scan"]
            same = cur is not None and cur[0] == new[0] and (
                new[1] == norm_tp(cur[1]) or new[1] == "default_or_day0" and cur[1] == "day0")
            if not same:
                st["scan"] = new
                st["segmented"], st["visible"], st["opacity"], st["focus"] = set(), {}, {}, None
        elif st["scan"] is None and name not in ("reset_view", "set_layout", "set_window", "export_measurements"):
            return None  # acting on a scan that isn't open
        elif name == "set_window":
            st["window"] = a["preset"]
        elif name in ("segment", "measure", "focus_on"):
            s = a["structure"]
            if s not in st["segmented"]:
                st["segmented"].add(s)
                st["visible"][s] = True
            if name == "measure":
                st["reported"].add(("measure", tuple(map(str, st["scan"])), s, a["metric"]))
            if name == "focus_on":
                st["focus"] = s
        elif name == "compare_timepoints":
            st["reported"].add(("compare", str(st["scan"][0]), a["structure"], a["metric"],
                                a["from_timepoint"], a["to_timepoint"]))
        elif name == "set_layout":
            st["layout"] = a["layout"]
        elif name == "set_visibility":
            st["visible"][a["structure"]] = bool(call["arguments"]["visible"])
        elif name == "set_opacity":
            o = a["opacity"]
            if not (0.0 <= o <= 1.0):
                return None
            st["opacity"][a["structure"]] = o
        elif name == "export_measurements":
            st["exports"].add(a["format"])
        elif name == "reset_view":
            st["layout"], st["window"], st["focus"] = "axial", "soft_tissue", None
    st["segmented"] = sorted(st["segmented"])
    st["visible"] = sorted(st["visible"].items())
    st["opacity"] = sorted(st["opacity"].items())
    st["reported"] = sorted(st["reported"])
    st["exports"] = sorted(st["exports"])
    return st


def outcome_match(pred, gold, state_key):
    sp = simulate(pred, state_key)
    return sp is not None and sp == simulate(gold, state_key)


# ---------------------------------------------------------------- running
def load_done(raw=RAW):
    done = {}
    if os.path.exists(raw):
        with open(raw) as f:
            for line in f:
                r = json.loads(line)
                done[(r["model"], r["mode"], r["case"])] = r
    return done


def run(models, modes, cases=CASES, raw=RAW):
    done = load_done(raw)
    os.makedirs(os.path.dirname(raw), exist_ok=True)
    with open(raw, "a") as out:
        for model in models:
            for mode in modes:
                if mode.startswith("native") and model in NO_NATIVE_TOOLS:
                    continue
                n_ok = 0
                for i, (cat, state, utt, gold) in enumerate(cases):
                    key = (model, mode, i)
                    if key in done:
                        n_ok += done[key]["outcome"]
                        continue
                    r = plan(utt, state, model, mode)
                    rec = {
                        "model": model, "mode": mode, "case": i, "category": cat,
                        "utterance": utt, "gold": gold, "pred": r["calls"],
                        "problems": r["problems"], "latency_s": round(r["latency_s"], 3),
                        "exact": exact_match(r["calls"], gold),
                        "outcome": outcome_match(r["calls"], gold, state),
                        "reply": r["reply"][:300],
                    }
                    n_ok += rec["outcome"]
                    out.write(json.dumps(rec) + "\n")
                    out.flush()
                print(f"{model:14s} {mode:10s} outcome {n_ok}/{len(cases)}", flush=True)


def summarize(raw=RAW, out_path=SUMMARY):
    done = load_done(raw)
    groups = defaultdict(list)
    for (model, mode, _), r in done.items():
        groups[(model, mode)].append(r)
    rows = []
    for (model, mode), rs in sorted(groups.items()):
        by_cat = defaultdict(list)
        for r in rs:
            by_cat[r["category"]].append(r["outcome"])
        lat = sorted(r["latency_s"] for r in rs)
        rows.append({
            "model": model, "mode": mode, "n": len(rs),
            "outcome_acc": round(sum(r["outcome"] for r in rs) / len(rs), 4),
            "exact_acc": round(sum(r["exact"] for r in rs) / len(rs), 4),
            "invalid_call_rate": round(sum(bool(r["problems"]) for r in rs) / len(rs), 4),
            "median_latency_s": lat[len(lat) // 2],
            "by_category": {k: round(sum(v) / len(v), 4) for k, v in sorted(by_cat.items())},
        })
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2)
    for r in rows:
        print(f"{r['model']:14s} {r['mode']:7s} n={r['n']:3d} outcome={r['outcome_acc']:.3f} "
              f"exact={r['exact_acc']:.3f} invalid={r['invalid_call_rate']:.3f} "
              f"p50={r['median_latency_s']:.2f}s {r['by_category']}")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--models", nargs="*", default=MODELS)
    ap.add_argument("--modes", nargs="*", default=MODES)
    ap.add_argument("--set", choices=list(SETS), default="main")
    args = ap.parse_args()
    cases, raw = SETS[args.set]
    if not args.summarize:
        run(args.models, args.modes, cases, raw)
    summarize(raw, SUMMARY if args.set == "main" else raw.replace(".jsonl", "_summary.json"))
