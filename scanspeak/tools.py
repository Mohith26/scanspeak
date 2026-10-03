"""Tool schema for ScanSpeak: the one place the agent's action space is defined.

Every backend (the browser/NiiVue backend and the 3D Slicer backend) implements
exactly these tools, and the benchmark scores models against exactly these
names and argument values. Keep this file small and boring on purpose: a 1.5B
model has to be able to read the whole thing in its prompt.
"""

STRUCTURES = ["tumor", "body", "bone", "lungs"]
WINDOW_PRESETS = ["soft_tissue", "bone", "lung", "full_range"]
LAYOUTS = ["axial", "coronal", "sagittal", "three_d", "four_up"]
METRICS = ["volume", "max_diameter", "mean_intensity"]


def _fn(name, description, properties, required):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


TOOLS = [
    _fn(
        "load_scan",
        "Open a mouse micro-CT scan. mouse_id looks like 'M03'. timepoint is the "
        "scan time label like 'day0', 'day7', '24h'; omit it for the first scan.",
        {
            "mouse_id": {"type": "string"},
            "timepoint": {"type": "string"},
        },
        ["mouse_id"],
    ),
    _fn(
        "set_window",
        "Change image contrast (window/level) to a named preset.",
        {"preset": {"type": "string", "enum": WINDOW_PRESETS}},
        ["preset"],
    ),
    _fn(
        "segment",
        "Automatically segment (outline in 3D) a structure in the open scan. "
        "'tumor' uses the trained AI model; others use intensity rules.",
        {"structure": {"type": "string", "enum": STRUCTURES}},
        ["structure"],
    ),
    _fn(
        "measure",
        "Measure a structure in the open scan. Segments it first if needed. "
        "volume is in mm^3, max_diameter in mm, mean_intensity in HU.",
        {
            "structure": {"type": "string", "enum": STRUCTURES},
            "metric": {"type": "string", "enum": METRICS},
        },
        ["structure", "metric"],
    ),
    _fn(
        "compare_timepoints",
        "Compare a structure's measurement for the same mouse between two "
        "timepoints and report absolute and percent change.",
        {
            "structure": {"type": "string", "enum": STRUCTURES},
            "metric": {"type": "string", "enum": METRICS},
            "from_timepoint": {"type": "string"},
            "to_timepoint": {"type": "string"},
        },
        ["structure", "metric", "from_timepoint", "to_timepoint"],
    ),
    _fn(
        "set_layout",
        "Change which views are shown.",
        {"layout": {"type": "string", "enum": LAYOUTS}},
        ["layout"],
    ),
    _fn(
        "focus_on",
        "Move the slice views so they are centered on a structure.",
        {"structure": {"type": "string", "enum": STRUCTURES}},
        ["structure"],
    ),
    _fn(
        "set_visibility",
        "Show or hide a structure's segmentation overlay.",
        {
            "structure": {"type": "string", "enum": STRUCTURES},
            "visible": {"type": "boolean"},
        },
        ["structure", "visible"],
    ),
    _fn(
        "set_opacity",
        "Set a structure overlay's opacity, from 0.0 (invisible) to 1.0 (solid).",
        {
            "structure": {"type": "string", "enum": STRUCTURES},
            "opacity": {"type": "number"},
        },
        ["structure", "opacity"],
    ),
    _fn(
        "export_measurements",
        "Save every measurement taken in this session to a file.",
        {"format": {"type": "string", "enum": ["csv", "json"]}},
        ["format"],
    ),
    _fn(
        "reset_view",
        "Reset zoom, pan, layout and contrast to defaults.",
        {},
        [],
    ),
]

TOOL_NAMES = [t["function"]["name"] for t in TOOLS]
TOOL_BY_NAME = {t["function"]["name"]: t for t in TOOLS}


def calls_json_schema():
    """JSON schema for grammar-constrained decoding.

    The model must emit {"calls": [...], "reply": "..."}; each call's name is
    restricted to the real tool names, so it can never invent a tool. Arguments
    are a loose object here and validated by validate_call() afterwards,
    because Ollama's grammar compiler handles one flat union far more reliably
    than eleven nested oneOf branches.
    """
    return {
        "type": "object",
        "properties": {
            "calls": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "enum": TOOL_NAMES},
                        "arguments": {"type": "object"},
                    },
                    "required": ["name", "arguments"],
                },
            },
            "reply": {"type": "string"},
        },
        "required": ["calls", "reply"],
    }


def strict_calls_json_schema():
    """Per-tool schema: each call is one of eleven exact shapes.

    Unlike calls_json_schema(), argument keys, enum values and types are all
    enforced by the decoder, so a typo like {"mouse_id: ": ...} is impossible.
    """
    variants = []
    for t in TOOLS:
        f = t["function"]
        params = f["parameters"]
        variants.append({
            "type": "object",
            "properties": {
                "name": {"type": "string", "enum": [f["name"]]},
                "arguments": {
                    "type": "object",
                    "properties": params["properties"],
                    "required": params["required"],
                    "additionalProperties": False,
                },
            },
            "required": ["name", "arguments"],
            "additionalProperties": False,
        })
    return {
        "type": "object",
        "properties": {
            "calls": {"type": "array", "items": {"anyOf": variants}},
            "reply": {"type": "string"},
        },
        "required": ["calls", "reply"],
    }


def repair_call(call):
    """Schema-driven type repair, applied after decoding (no model involved).

    Small models in native tool-calling mode often get the right tool and value
    but the wrong JSON type: "False" instead of false, "0.3" instead of 0.3,
    "30%" for an opacity. This coerces values to the type the schema declares
    and drops nothing else. It never invents a value that wasn't emitted.
    """
    name = call.get("name")
    if name not in TOOL_BY_NAME:
        return call
    props = TOOL_BY_NAME[name]["function"]["parameters"]["properties"]
    args = dict(call.get("arguments") or {})
    for k, v in list(args.items()):
        spec = props.get(k)
        if spec is None or not isinstance(v, str):
            continue
        s = v.strip().lower()
        if spec["type"] == "boolean" and s in ("true", "false"):
            args[k] = s == "true"
        elif spec["type"] == "number":
            try:
                args[k] = float(s[:-1]) / 100 if s.endswith("%") else float(s)
            except ValueError:
                pass
        elif "enum" in spec:
            cand = s.replace(" ", "_").replace("-", "_")
            if cand in spec["enum"]:
                args[k] = cand
    return {"name": name, "arguments": args}


def validate_call(call):
    """Return a list of human-readable problems with one tool call ([] = ok)."""
    problems = []
    name = call.get("name")
    if name not in TOOL_BY_NAME:
        return [f"unknown tool {name!r}"]
    params = TOOL_BY_NAME[name]["function"]["parameters"]
    args = call.get("arguments") or {}
    for req in params["required"]:
        if req not in args:
            problems.append(f"{name}: missing {req}")
    for key, val in args.items():
        spec = params["properties"].get(key)
        if spec is None:
            problems.append(f"{name}: unexpected argument {key}")
            continue
        if "enum" in spec and val not in spec["enum"]:
            problems.append(f"{name}: {key}={val!r} not in {spec['enum']}")
        if spec["type"] == "number" and not isinstance(val, (int, float)):
            problems.append(f"{name}: {key} should be a number")
        if spec["type"] == "boolean" and not isinstance(val, bool):
            problems.append(f"{name}: {key} should be true/false")
    return problems


def tools_as_text():
    """Compact plain-text tool list for prompt-only and schema modes."""
    lines = []
    for t in TOOLS:
        f = t["function"]
        props = f["parameters"]["properties"]
        req = set(f["parameters"]["required"])
        args = []
        for k, spec in props.items():
            typ = "|".join(spec["enum"]) if "enum" in spec else spec["type"]
            args.append(f"{k}{'' if k in req else '?'}: {typ}")
        lines.append(f"- {f['name']}({', '.join(args)}): {f['description']}")
    return "\n".join(lines)
