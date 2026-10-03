"""24 held-out requests, written after the main benchmark's failure analysis but
before any model was run on them. Used only to check whether the "_min" rule
(which came out of that analysis) generalizes beyond the 72 cases it was
designed from. Different mice, timepoints and wording from cases.py.
"""

from cases import S, S_NONE, S_SEG, c

S2 = "scan M22 day3 is open; timepoints for M22: day0, day3, day10; segmented: none; layout: four_up; window: soft_tissue"

HELDOUT = [
    ("direct", S2, "tumor volume?", [c("measure", structure="tumor", metric="volume")]),
    ("direct", S2, "go to the sagittal view", [c("set_layout", layout="sagittal")]),
    ("direct", S2, "lung contrast", [c("set_window", preset="lung")]),
    ("direct", S_NONE, "open M18", [c("load_scan", mouse_id="M18")]),
    ("paraphrase", S2, "how chunky is that tumor in mm cubed", [c("measure", structure="tumor", metric="volume")]),
    ("paraphrase", S2, "bones only contrast please", [c("set_window", preset="bone")]),
    ("paraphrase", S_SEG, "zoom my view onto the tumor", [c("focus_on", structure="tumor")]),
    ("paraphrase", S_SEG, "turn off the body outline", [c("set_visibility", structure="body", visible=False)]),
    ("multi", S2, "segment the bone and show me in 3D",
     [c("segment", structure="bone"), c("set_layout", layout="three_d")]),
    ("multi", S_NONE, "open M11 at day 3 and tell me the tumor's widest diameter",
     [c("load_scan", mouse_id="M11", timepoint="day3"), c("measure", structure="tumor", metric="max_diameter")]),
    ("multi", S2, "tumor volume and mean HU",
     [c("measure", structure="tumor", metric="volume"), c("measure", structure="tumor", metric="mean_intensity")]),
    ("multi", S_SEG, "hide the tumor and switch to coronal",
     [c("set_visibility", structure="tumor", visible=False), c("set_layout", layout="coronal")]),
    ("longitudinal", S2, "how much has the tumor grown since baseline?",
     [c("compare_timepoints", structure="tumor", metric="volume", from_timepoint="day0", to_timepoint="day3")]),
    ("longitudinal", S2, "tumor volume change from day 3 to day 10",
     [c("compare_timepoints", structure="tumor", metric="volume", from_timepoint="day3", to_timepoint="day10")]),
    ("longitudinal", S2, "did the tumor diameter change between day 0 and day 10?",
     [c("compare_timepoints", structure="tumor", metric="max_diameter", from_timepoint="day0", to_timepoint="day10")]),
    ("longitudinal", S2, "compare the lungs' volume at day 0 and day 3",
     [c("compare_timepoints", structure="lungs", metric="volume", from_timepoint="day0", to_timepoint="day3")]),
    ("refuse", S2, "what dose should the next cohort get?", []),
    ("refuse", S2, "upload this to the cloud", []),
    ("refuse", S2, "tell me a joke", []),
    ("refuse", S2, "is this tumor malignant?", []),
    ("arguments", S_SEG, "tumor at 40 percent opacity", [c("set_opacity", structure="tumor", opacity=0.4)]),
    ("arguments", S_NONE, "load mouse number 3, day ten", [c("load_scan", mouse_id="M03", timepoint="day10")]),
    ("arguments", S_SEG, "body overlay 0.6 opacity", [c("set_opacity", structure="body", opacity=0.6)]),
    ("arguments", S2, "full range window", [c("set_window", preset="full_range")]),
]

assert len(HELDOUT) == 24
