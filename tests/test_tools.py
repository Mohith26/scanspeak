"""Run: python3 tests/test_tools.py (no pytest needed)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bench"))
from scanspeak.tools import repair_call, validate_call, strict_calls_json_schema, TOOL_NAMES
from run_bench import outcome_match, exact_match, CASES, HELDOUT
from cases import c, S, S_SEG

# validation
assert validate_call(c("set_window", preset="bone")) == []
assert validate_call(c("set_window", preset="Bone"))            # enum is case-sensitive
assert validate_call(c("nope"))                                  # unknown tool
assert validate_call(c("set_opacity", structure="tumor", opacity="0.3"))
# repair fixes types, never invents values
assert repair_call(c("set_opacity", structure="tumor", opacity="0.3"))["arguments"]["opacity"] == 0.3
assert repair_call(c("set_opacity", structure="tumor", opacity="30%"))["arguments"]["opacity"] == 0.3
assert repair_call(c("set_visibility", structure="body", visible="False"))["arguments"]["visible"] is False
assert repair_call(c("set_window", preset="soft tissue"))["arguments"]["preset"] == "soft_tissue"
assert repair_call(c("measure", structure="tumor"))["arguments"] == {"structure": "tumor"}
assert validate_call(repair_call(c("set_opacity", structure="tumor", opacity="banana")))
# strict schema has one branch per tool
assert len(strict_calls_json_schema()["properties"]["calls"]["items"]["anyOf"]) == len(TOOL_NAMES)
# every gold answer passes its own scorer
for cat, st, u, g in CASES + HELDOUT:
    assert exact_match(g, g) and outcome_match(g, g, st), u
# outcome scoring forgives harmless extras, punishes side effects
assert outcome_match([c("segment", structure="tumor"), c("measure", structure="tumor", metric="volume")],
                     [c("measure", structure="tumor", metric="volume")], S)
assert not outcome_match([c("export_measurements", format="csv")], [], S)
assert not outcome_match([c("set_opacity", structure="tumor", opacity=50)], [c("set_opacity", structure="tumor", opacity=0.5)], S_SEG)
print("tools/scoring tests passed")

# ---- catalog / grounding (needs scanspeak/data/catalog.json, no scans)
from scanspeak.backends.volume_backend import (ANIMALS, animals_in_text, resolve_animal,
                                               snap_tp, timepoints_in_text)
assert len(ANIMALS) == 223 and sum(a["held_out"] for a in ANIMALS.values()) == 55
assert resolve_animal("D8-M37") == ("D8-M37", None)
assert resolve_animal("dataset 8 mouse 37")[0] == "D8-M37"
assert resolve_animal("M37")[0] is None and "D10-M37" in resolve_animal("M37")[1]   # ambiguous -> ask
assert resolve_animal("D99-M01")[0] is None
assert timepoints_in_text("open D1-M01 at 22.5 hours") == ["22.5h"]
assert timepoints_in_text("compare day 0 to day 13") == ["day0", "day13"]
assert animals_in_text("open dataset 4 mouse 10 at 16h") == ["D4-M10"]
assert snap_tp("22h", ["0h", "3h", "22.5h"]) == "22.5h" and snap_tp("10h", ["0h", "3h", "22.5h"]) is None
print("catalog/grounding tests passed")
