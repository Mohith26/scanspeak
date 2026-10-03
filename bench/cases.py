"""ScanSpeak benchmark: 72 requests a preclinical imaging scientist might type.

Each case has a gold list of tool calls. Scoring (see run_bench.py) is done two
ways: exact call-sequence match, and outcome match, where both the model's
calls and the gold calls are executed on a tiny simulated viewer and we check
the final viewer state plus the set of numbers reported to the user. Outcome
match forgives harmless differences (re-opening the scan that is already open,
segmenting before measuring) but not wrong numbers or wrong views.

Written by hand before any model was run against it; not tuned afterwards.
"""

S = "scan M07 day7 is open; timepoints for M07: day0, day7, day14, day21; segmented: none; layout: axial; window: soft_tissue"
S_SEG = "scan M07 day7 is open; timepoints for M07: day0, day7, day14, day21; segmented: tumor, body; layout: four_up; window: soft_tissue"
S_NONE = "no scan is open; available mice: M01-M40"


def c(name, **arguments):
    return {"name": name, "arguments": arguments}


CASES = [
    # ---- direct: one explicit tool --------------------------------------
    ("direct", S_NONE, "Load mouse M12", [c("load_scan", mouse_id="M12")]),
    ("direct", S_NONE, "open M03 at day 14", [c("load_scan", mouse_id="M03", timepoint="day14")]),
    ("direct", S, "Set the window to bone", [c("set_window", preset="bone")]),
    ("direct", S, "switch to the lung window preset", [c("set_window", preset="lung")]),
    ("direct", S, "segment the tumor", [c("segment", structure="tumor")]),
    ("direct", S, "Segment the lungs", [c("segment", structure="lungs")]),
    ("direct", S, "what is the tumor volume?", [c("measure", structure="tumor", metric="volume")]),
    ("direct", S, "measure the max diameter of the tumor", [c("measure", structure="tumor", metric="max_diameter")]),
    ("direct", S, "mean intensity of the tumor please", [c("measure", structure="tumor", metric="mean_intensity")]),
    ("direct", S, "show the coronal view", [c("set_layout", layout="coronal")]),
    ("direct", S, "switch to the four-up layout", [c("set_layout", layout="four_up")]),
    ("direct", S, "3D view", [c("set_layout", layout="three_d")]),
    ("direct", S_SEG, "center the views on the tumor", [c("focus_on", structure="tumor")]),
    ("direct", S_SEG, "hide the body segmentation", [c("set_visibility", structure="body", visible=False)]),
    ("direct", S_SEG, "show the tumor overlay again", [c("set_visibility", structure="tumor", visible=True)]),
    ("direct", S_SEG, "set tumor opacity to 0.3", [c("set_opacity", structure="tumor", opacity=0.3)]),
    ("direct", S_SEG, "export the measurements as csv", [c("export_measurements", format="csv")]),
    ("direct", S_SEG, "save measurements to json", [c("export_measurements", format="json")]),
    ("direct", S_SEG, "reset the view", [c("reset_view")]),
    ("direct", S, "how big is the bone volume", [c("measure", structure="bone", metric="volume")]),

    # ---- paraphrase: biologist / lay wording -----------------------------
    ("paraphrase", S, "make the skeleton pop out", [c("set_window", preset="bone")]),
    ("paraphrase", S, "how large is the tumor in cubic millimeters", [c("measure", structure="tumor", metric="volume")]),
    ("paraphrase", S, "how wide is the tumor at its widest point", [c("measure", structure="tumor", metric="max_diameter")]),
    ("paraphrase", S_SEG, "get rid of the tumor outline, it's in the way", [c("set_visibility", structure="tumor", visible=False)]),
    ("paraphrase", S_SEG, "make the tumor see-through, like half transparent", [c("set_opacity", structure="tumor", opacity=0.5)]),
    ("paraphrase", S_SEG, "take me to the tumor", [c("focus_on", structure="tumor")]),
    ("paraphrase", S, "I want to see it from the side", [c("set_layout", layout="sagittal")]),
    ("paraphrase", S, "top-down slices please", [c("set_layout", layout="axial")]),
    ("paraphrase", S, "give me a 3D rendering", [c("set_layout", layout="three_d")]),
    ("paraphrase", S, "outline the whole mouse", [c("segment", structure="body")]),
    ("paraphrase", S_SEG, "undo all my zooming and contrast changes", [c("reset_view")]),
    ("paraphrase", S_SEG, "I need these numbers in a spreadsheet", [c("export_measurements", format="csv")]),
    ("paraphrase", S, "find the tumor", [c("segment", structure="tumor")]),
    ("paraphrase", S, "how dense is the tumor tissue (HU)?", [c("measure", structure="tumor", metric="mean_intensity")]),
    ("paraphrase", S_SEG, "make the body overlay barely visible, like 10 percent", [c("set_opacity", structure="body", opacity=0.1)]),

    # ---- multi-step -------------------------------------------------------
    ("multi", S_NONE, "open M21 day 7 and measure the tumor volume",
     [c("load_scan", mouse_id="M21", timepoint="day7"), c("measure", structure="tumor", metric="volume")]),
    ("multi", S, "segment the tumor and then center on it",
     [c("segment", structure="tumor"), c("focus_on", structure="tumor")]),
    ("multi", S, "bone window and four up layout",
     [c("set_window", preset="bone"), c("set_layout", layout="four_up")]),
    ("multi", S, "measure tumor volume and max diameter",
     [c("measure", structure="tumor", metric="volume"), c("measure", structure="tumor", metric="max_diameter")]),
    ("multi", S, "segment the lungs, switch to the lung window, and show it in 3D",
     [c("segment", structure="lungs"), c("set_window", preset="lung"), c("set_layout", layout="three_d")]),
    ("multi", S_SEG, "hide the body, focus on the tumor, and make the tumor 70% opaque",
     [c("set_visibility", structure="body", visible=False), c("focus_on", structure="tumor"),
      c("set_opacity", structure="tumor", opacity=0.7)]),
    ("multi", S, "get the tumor volume then export everything as json",
     [c("measure", structure="tumor", metric="volume"), c("export_measurements", format="json")]),
    ("multi", S_NONE, "load M05 at day 21, segment the tumor, and give me its diameter",
     [c("load_scan", mouse_id="M05", timepoint="day21"), c("segment", structure="tumor"),
      c("measure", structure="tumor", metric="max_diameter")]),
    ("multi", S, "measure the bone volume and the body volume",
     [c("measure", structure="bone", metric="volume"), c("measure", structure="body", metric="volume")]),
    ("multi", S_SEG, "reset everything and then go to the coronal view",
     [c("reset_view"), c("set_layout", layout="coronal")]),
    ("multi", S_NONE, "open M30 and switch to bone contrast",
     [c("load_scan", mouse_id="M30"), c("set_window", preset="bone")]),
    ("multi", S, "segment the tumor, measure its volume, and show it in 3D",
     [c("segment", structure="tumor"), c("measure", structure="tumor", metric="volume"),
      c("set_layout", layout="three_d")]),
    ("multi", S_SEG, "show the body overlay at 20% opacity",
     [c("set_visibility", structure="body", visible=True), c("set_opacity", structure="body", opacity=0.2)]),
    ("multi", S, "full range window then sagittal",
     [c("set_window", preset="full_range"), c("set_layout", layout="sagittal")]),
    ("multi", S_NONE, "open mouse M02 at 24h, measure the tumor's mean intensity",
     [c("load_scan", mouse_id="M02", timepoint="24h"), c("measure", structure="tumor", metric="mean_intensity")]),

    # ---- longitudinal: comparisons across timepoints ----------------------
    ("longitudinal", S, "how much did the tumor grow since day 0?",
     [c("compare_timepoints", structure="tumor", metric="volume", from_timepoint="day0", to_timepoint="day7")]),
    ("longitudinal", S, "compare tumor volume between day 7 and day 21",
     [c("compare_timepoints", structure="tumor", metric="volume", from_timepoint="day7", to_timepoint="day21")]),
    ("longitudinal", S, "did the tumor shrink from baseline to the last scan?",
     [c("compare_timepoints", structure="tumor", metric="volume", from_timepoint="day0", to_timepoint="day21")]),
    ("longitudinal", S, "change in tumor diameter from day 0 to day 14",
     [c("compare_timepoints", structure="tumor", metric="max_diameter", from_timepoint="day0", to_timepoint="day14")]),
    ("longitudinal", S, "what's the percent change in tumor volume between the first and second scans",
     [c("compare_timepoints", structure="tumor", metric="volume", from_timepoint="day0", to_timepoint="day7")]),
    ("longitudinal", S, "track the tumor volume from day 14 to day 21",
     [c("compare_timepoints", structure="tumor", metric="volume", from_timepoint="day14", to_timepoint="day21")]),
    ("longitudinal", S, "has the bone volume changed since day 0",
     [c("compare_timepoints", structure="bone", metric="volume", from_timepoint="day0", to_timepoint="day7")]),
    ("longitudinal", S, "compare tumor volume day0 vs day21 and show it in 3D",
     [c("compare_timepoints", structure="tumor", metric="volume", from_timepoint="day0", to_timepoint="day21"),
      c("set_layout", layout="three_d")]),

    # ---- refuse: out of scope, correct answer is no tool calls ------------
    ("refuse", S, "is this mouse going to survive the study?", []),
    ("refuse", S, "should we increase the drug dose for this group?", []),
    ("refuse", S, "diagnose what kind of cancer this is", []),
    ("refuse", S, "write me a poem about mice", []),
    ("refuse", S, "email these results to my PI", []),
    ("refuse", S, "what's the weather in Durham", []),
    ("refuse", S, "delete all the scans from the server", []),
    ("refuse", S, "is the treatment working? give me a yes or no", []),

    # ---- tricky arguments --------------------------------------------------
    ("arguments", S_SEG, "tumor opacity 100%", [c("set_opacity", structure="tumor", opacity=1.0)]),
    ("arguments", S_SEG, "make the tumor fully invisible but keep it turned on, opacity zero",
     [c("set_opacity", structure="tumor", opacity=0.0)]),
    ("arguments", S_NONE, "load mouse 9 at day seven", [c("load_scan", mouse_id="M09", timepoint="day7")]),
    ("arguments", S_NONE, "pull up m14, the 48 hour scan", [c("load_scan", mouse_id="M14", timepoint="48h")]),
    ("arguments", S, "soft tissue contrast please", [c("set_window", preset="soft_tissue")]),
    ("arguments", S_SEG, "quarter opacity on the body", [c("set_opacity", structure="body", opacity=0.25)]),
]

assert len(CASES) == 72, len(CASES)
