"""Turn one plain-English request into a list of validated tool calls.

Three decoding modes, all talking to a local Ollama server:

  native  - Ollama's built-in tool calling (model must support "tools").
  schema  - grammar-constrained decoding: the model is forced to emit JSON that
            matches calls_json_schema(), so tool names are always real.
  strict  - like schema, but every tool's argument names, enums and types are
            also enforced by the decoder (one anyOf branch per tool).
  prompt  - same prompt as "schema" but no constraint; we just try to parse
            JSON out of whatever text comes back. This is the naive baseline.

Append "_min" to any mode (e.g. "strict_min") to add one extra rule telling the
model to emit only the calls the request needs. That rule came out of the
benchmark's failure analysis, so it is evaluated on a separate held-out set.

Design rule: the language model only ever chooses actions. Every number the
user sees comes from a backend tool, never from model text, so a small model
cannot hallucinate a tumor volume.
"""

import json
import re
import time
import urllib.request

from .tools import (TOOLS, calls_json_schema, repair_call, strict_calls_json_schema,
                    tools_as_text, validate_call)

OLLAMA_URL = "http://localhost:11434/api/chat"

SYSTEM_TEMPLATE = """You control a preclinical micro-CT viewer for mouse imaging studies.
Translate the user's request into tool calls. Use only these tools:

{tools}

Rules:
- Emit the calls in the order they should run.
- Use the exact enum values listed (for example "soft_tissue", not "soft tissue").
- Timepoints look like "day0", "day7", "24h". Mouse ids look like "M03".
- If the request is not something these tools can do (for example a diagnosis,
  a treatment decision, or anything unrelated to the viewer), return no calls
  and explain briefly in "reply".
- Never make up measurements; measuring is done by the tools.
{extra}
Current viewer state: {state}
"""

JSON_INSTRUCTIONS = """
Respond with JSON only, in this form:
{"calls": [{"name": "<tool>", "arguments": {...}}], "reply": "<one short sentence>"}"""


def _post(payload, timeout=180):
    req = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def _extract_json(text):
    """Best-effort: pull the first balanced {...} object out of free text."""
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None


def _coerce_args(args):
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            return {}
    return args if isinstance(args, dict) else {}


MINIMAL_RULE = ("- Include only the calls needed for exactly what was asked. Do not add extra "
                "measurements, exports, or a reload of the scan that is already open; "
                "measure and compare_timepoints segment automatically.\n")


def plan(utterance, state_text, model, mode="schema", temperature=0.0, seed=0, repair=False):
    """Return a dict: calls, reply, problems, latency_s, raw.

    repair=True applies schema-driven type repair (tools.repair_call) to the
    decoded calls before validation. The benchmark stores unrepaired outputs and
    scores repair offline, so both numbers come from the same model responses.
    """
    extra = ""
    if mode.endswith("_min"):
        mode, extra = mode[:-4], MINIMAL_RULE
    system = SYSTEM_TEMPLATE.format(tools=tools_as_text(), state=state_text, extra=extra)
    payload = {
        "model": model,
        "stream": False,
        "options": {"temperature": temperature, "seed": seed, "num_ctx": 4096},
        "keep_alive": "30m",
    }
    if mode == "native":
        payload["messages"] = [
            {"role": "system", "content": system},
            {"role": "user", "content": utterance},
        ]
        payload["tools"] = TOOLS
    elif mode in ("schema", "strict", "prompt"):
        payload["messages"] = [
            {"role": "system", "content": system + JSON_INSTRUCTIONS},
            {"role": "user", "content": utterance},
        ]
        if mode == "schema":
            payload["format"] = calls_json_schema()
        elif mode == "strict":
            payload["format"] = strict_calls_json_schema()
    else:
        raise ValueError(mode)

    t0 = time.time()
    try:
        out = _post(payload)
    except Exception as exc:  # network / model errors are scored as failures
        return {"calls": [], "reply": "", "problems": [f"error: {exc}"],
                "latency_s": time.time() - t0, "raw": ""}
    latency = time.time() - t0
    msg = out.get("message", {})
    raw = msg.get("content", "") or ""

    calls, reply = [], ""
    if mode == "native":
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function", {})
            calls.append({"name": fn.get("name"),
                          "arguments": _coerce_args(fn.get("arguments"))})
        reply = raw.strip()
        # Some models ignore the tools channel and print JSON in content.
        if not calls and raw:
            parsed = _extract_json(raw)
            if isinstance(parsed, dict) and "calls" in parsed:
                calls = parsed.get("calls") or []
            elif isinstance(parsed, dict) and "name" in parsed:
                calls = [{"name": parsed["name"],
                          "arguments": _coerce_args(parsed.get("arguments") or parsed.get("parameters"))}]
    else:
        parsed = _extract_json(raw)
        if isinstance(parsed, dict):
            calls = parsed.get("calls") or []
            reply = parsed.get("reply", "")

    clean = []
    for c in calls if isinstance(calls, list) else []:
        if isinstance(c, dict):
            clean.append({"name": c.get("name"),
                          "arguments": _coerce_args(c.get("arguments"))})
    if repair:
        clean = [repair_call(c) for c in clean]
    problems = [p for c in clean for p in validate_call(c)]
    return {"calls": clean, "reply": reply, "problems": problems,
            "latency_s": latency, "raw": raw}


def available_models():
    req = urllib.request.Request("http://localhost:11434/api/tags")
    with urllib.request.urlopen(req, timeout=10) as resp:
        return [m["name"] for m in json.loads(resp.read())["models"]]


_WS = re.compile(r"\s+")


def normalize(value):
    """Normalize argument values for scoring: case/space-insensitive strings."""
    if isinstance(value, str):
        v = _WS.sub("", value.strip().lower())
        v = v.replace("day_", "day").replace("hours", "h").replace("hour", "h")
        return v
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value
