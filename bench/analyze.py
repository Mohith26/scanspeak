"""Failure taxonomy + figures for the ScanSpeak benchmark.

    python3 bench/analyze.py

Every failed case is put in exactly one bucket:
  invalid       a call fails schema validation (bad tool/argument/enum/type)
  false_action  gold is "do nothing" (out of scope) but the model acted
  false_refusal gold has calls but the model returned none
  over_call     every gold call is present, plus extra unrequested calls
  under_call    the model's calls are a strict subset of the gold calls
  wrong         anything else (wrong tool, wrong argument value, wrong mouse...)
"""

import json
import os
import sys
from collections import Counter, defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
from run_bench import norm_args  # noqa: E402

MODEL_ORDER = ["qwen2.5:1.5b", "qwen2.5:3b", "llama3.2:3b", "gemma3:4b", "llama3.1:8b"]
MODE_ORDER = ["prompt", "schema", "strict", "native", "native+repair"]
MODE_COL = {"prompt": "#5a6275", "schema": "#8b93a7", "strict": "#4fd1c5", "native": "#f6ad55",
            "strict_min": "#2c7a7b", "native_min": "#c05621", "native+repair": "#c05621"}
BUCKETS = ["invalid", "false_action", "false_refusal", "over_call", "under_call", "wrong"]
BUCKET_COL = ["#fc8181", "#b794f4", "#90cdf4", "#f6ad55", "#68d391", "#a0aec0"]


def key(call):
    return (call.get("name"), json.dumps(norm_args(call), sort_keys=True))


def bucket(r):
    if r["outcome"]:
        return None
    if r["problems"]:
        return "invalid"
    gold, pred = Counter(map(key, r["gold"])), Counter(map(key, r["pred"]))
    if not r["gold"]:
        return "false_action"
    if not r["pred"]:
        return "false_refusal"
    if not (gold - pred) and (pred - gold):
        return "over_call"
    if not (pred - gold) and (gold - pred):
        return "under_call"
    return "wrong"


def load(path):
    with open(path) as f:
        return [json.loads(l) for l in f]


def table(rows):
    g = defaultdict(list)
    for r in rows:
        g[(r["model"], r["mode"])].append(r)
    out = {}
    for k, rs in g.items():
        b = Counter(bucket(r) for r in rs)
        out[k] = {"n": len(rs), "acc": sum(r["outcome"] for r in rs) / len(rs),
                  "buckets": {x: b.get(x, 0) for x in BUCKETS}}
    return out


def main():
    main_rows = load(os.path.join(ROOT, "results", "raw.jsonl"))
    t = table(main_rows)
    for k, v in rescore_with_repair(os.path.join(ROOT, "results", "raw.jsonl")).items():
        m, mo = k.split("|")
        if mo == "native":
            t[(m, "native+repair")] = {"n": v["n"], "acc": v["acc"], "buckets": {}}
    report = {"main": {f"{m}|{mo}": v for (m, mo), v in t.items()}}

    held_path = os.path.join(ROOT, "results", "raw_heldout.jsonl")
    if os.path.exists(held_path):
        th = table(load(held_path))
        report["heldout"] = {f"{m}|{mo}": v for (m, mo), v in th.items()}

    # aggregate over models for the headline
    agg = defaultdict(lambda: Counter())
    acc = defaultdict(list)
    for (m, mo), v in t.items():
        if m == "gemma3:4b" or mo == "native+repair":  # same 4 tool-capable models in every mode
            continue
        agg[mo].update(v["buckets"])
        acc[mo].append(v["acc"])
    report["by_mode_excl_gemma"] = {mo: {"mean_acc": sum(a) / len(a), "buckets": dict(agg[mo])}
                                    for mo, a in acc.items()}
    with open(os.path.join(ROOT, "results", "analysis.json"), "w") as f:
        json.dump(report, f, indent=1)

    for mo, v in report["by_mode_excl_gemma"].items():
        print(f"{mo:11s} mean acc {v['mean_acc']:.3f}  {v['buckets']}")
    if "heldout" in report:
        print("held-out:")
        for k, v in sorted(report["heldout"].items()):
            print(f"  {k:28s} {v['acc']:.3f} {v['buckets']}")

    # ---- figure 1: accuracy grid
    os.makedirs(os.path.join(ROOT, "docs"), exist_ok=True)
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.6), gridspec_kw={"width_ratios": [1.5, 1]})
    w = 0.16
    for j, mo in enumerate(MODE_ORDER):
        xs, ys = [], []
        for i, m in enumerate(MODEL_ORDER):
            if (m, mo) in t:
                xs.append(i + (j - 2) * w)
                ys.append(t[(m, mo)]["acc"] * 100)
        ax[0].bar(xs, ys, w, color=MODE_COL[mo], label={"prompt": "prompt only", "schema": "loose JSON schema",
                                                         "strict": "strict per-tool schema", "native": "native tool calling",
                                                         "native+repair": "native + schema repair"}[mo])
    ax[0].set_xticks(range(len(MODEL_ORDER)))
    ax[0].set_xticklabels(MODEL_ORDER)
    ax[0].set_ylabel("requests handled correctly (%)")
    ax[0].set_ylim(0, 100)
    ax[0].legend(frameon=False, ncol=2, fontsize=9, loc="upper left")
    ax[0].set_title("72 imaging requests, outcome-scored", fontsize=12)
    ax[0].grid(axis="y", alpha=0.25)

    # ---- figure 1b: failure taxonomy per mode (pooled over 4 tool-capable models)
    modes = [m for m in ["prompt", "schema", "strict", "native"] if m in report["by_mode_excl_gemma"]]
    bottom = [0] * len(modes)
    for b, col in zip(BUCKETS, BUCKET_COL):
        vals = [report["by_mode_excl_gemma"][mo]["buckets"].get(b, 0) for mo in modes]
        ax[1].bar(modes, vals, bottom=bottom, color=col, label=b.replace("_", " "))
        bottom = [x + y for x, y in zip(bottom, vals)]
    ax[1].set_ylabel("failed requests (4 models x 72)")
    ax[1].set_title("Why requests fail", fontsize=12)
    ax[1].legend(frameon=False, fontsize=9)
    ax[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "docs", "benchmark.png"), dpi=150, bbox_inches="tight")
    print("wrote docs/benchmark.png")




def rescore_with_repair(path):
    """Offline: apply tools.repair_call to stored raw predictions and re-score."""
    from run_bench import INITIAL, exact_match, outcome_match  # noqa: F401
    from scanspeak.tools import repair_call, validate_call
    from cases import CASES
    from heldout import HELDOUT
    cases = HELDOUT if "heldout" in path else CASES
    out = defaultdict(lambda: [0, 0, 0])
    for r in load(path):
        pred = [repair_call(c) for c in r["pred"]]
        state = cases[r["case"]][1]
        ok = outcome_match(pred, r["gold"], state)
        inv = any(validate_call(c) for c in pred)
        o = out[(r["model"], r["mode"])]
        o[0] += ok
        o[1] += 1
        o[2] += inv
    return {f"{m}|{mo}": {"acc": a / n, "invalid_rate": i / n, "n": n} for (m, mo), (a, n, i) in out.items()}


if __name__ == "__main__" and "--repair" in sys.argv:
    for name in ["raw.jsonl", "raw_heldout.jsonl"]:
        p = os.path.join(ROOT, "results", name)
        if os.path.exists(p):
            res = rescore_with_repair(p)
            json.dump(res, open(p.replace(".jsonl", "_repaired.json"), "w"), indent=1)
            print(name)
            for k, v in sorted(res.items()):
                print(f"  {k:28s} acc {v['acc']:.3f} invalid {v['invalid_rate']:.3f}")


if __name__ == "__main__" and "--repair" not in sys.argv:
    main()
