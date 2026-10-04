"""Local web app: NiiVue viewer + chat, driven by a small local LLM.

    SCANSPEAK_TUMOR_MODEL=/path/to/model.pt python3 -m scanspeak.server --port 8417

Then open http://localhost:8765. Requires Ollama running locally.
"""

import argparse
import json
import mimetypes
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .agent import available_models, plan
from .backends.volume_backend import ANIMALS, CACHE, VolumeBackend

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEB = os.path.join(ROOT, "web")
BACKEND = VolumeBackend()
LOCK = threading.Lock()
DEFAULT_MODEL = os.environ.get("SCANSPEAK_MODEL", "llama3.1:8b")
DEFAULT_MODE = os.environ.get("SCANSPEAK_MODE", "native+repair")


def chat(text, model, mode):
    with LOCK:
        state = BACKEND.state_text()
        repair = mode.endswith("+repair")
        p = plan(text, state, model, mode.replace("+repair", ""), repair=repair)
        t0 = time.time()
        results = BACKEND.execute(p["calls"], utterance=text) if not p["problems"] else []
        exec_s = time.time() - t0
        if p["problems"]:
            reply = "I couldn't turn that into a valid action (" + "; ".join(p["problems"][:2]) + ")."
        elif not p["calls"]:
            reply = p["reply"] or "That isn't something this viewer can do."
        else:
            reply = " ".join(r["text"] for r in results)
        return {
            "calls": p["calls"], "results": [{k: v for k, v in r.items() if k != "call"} for r in results],
            "ops": [op for r in results for op in r.get("ops", [])],
            "reply": reply, "model_reply": p["reply"], "problems": p["problems"],
            "llm_s": round(p["latency_s"], 2), "exec_s": round(exec_s, 2),
            "state": BACKEND.state_text(), "measurements": BACKEND.measurements[-12:],
        }


def direct(calls):
    """Run tool calls picked in the UI (e.g. the animal list) without the language model."""
    with LOCK:
        t0 = time.time()
        results = BACKEND.execute(calls)
        return {
            "calls": calls, "results": [{k: v for k, v in r.items() if k != "call"} for r in results],
            "ops": [op for r in results for op in r.get("ops", [])],
            "reply": " ".join(r["text"] for r in results), "model_reply": "", "problems": [],
            "llm_s": 0, "exec_s": round(time.time() - t0, 2), "direct": True,
            "state": BACKEND.state_text(), "measurements": BACKEND.measurements[-12:],
        }


CATALOG_VIEW = [{"id": a["id"], "dataset": a["dataset"], "mouse": a["mouse"], "held_out": a["held_out"],
                 "scans": [{"tp": sc["tp"], "expert": sc["expert_tumor_mm3"]} for sc in a["scans"]]}
                for a in ANIMALS.values()]


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _file(self, base, rel):
        path = os.path.realpath(os.path.join(base, rel))
        if not path.startswith(os.path.realpath(base)) and not os.path.islink(os.path.join(base, rel)):
            return self._send(403, {"error": "forbidden"})
        if not os.path.exists(path):
            return self._send(404, {"error": "not found"})
        ctype = "application/octet-stream" if path.endswith(".gz") else (
            mimetypes.guess_type(path)[0] or "application/octet-stream")
        with open(path, "rb") as f:
            self._send(200, f.read(), ctype)

    def do_GET(self):
        url = self.path.split("?")[0]
        if url == "/api/info":
            try:
                models = available_models()
            except Exception:
                models = []
            return self._send(200, {"models": models, "model": DEFAULT_MODEL, "mode": DEFAULT_MODE,
                                    "state": BACKEND.state_text(),
                                    "tumor_model": BACKEND.tumor_model is not None})
        if url == "/api/catalog":
            return self._send(200, {"animals": CATALOG_VIEW})
        if url.startswith("/cache/"):
            return self._file(CACHE, url[len("/cache/"):])
        return self._file(WEB, "index.html" if url == "/" else url.lstrip("/"))

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        if self.path == "/api/chat":
            return self._send(200, chat(body.get("text", ""), body.get("model", DEFAULT_MODEL),
                                        body.get("mode", DEFAULT_MODE)))
        if self.path == "/api/exec":
            return self._send(200, direct(body.get("calls", [])))
        if self.path == "/api/reset":  # fresh session per page load
            global BACKEND
            with LOCK:
                BACKEND = VolumeBackend()
            return self._send(200, {"ok": True})
        if self.path == "/api/shutdown":  # sandbox-friendly: lets a test harness free the port
            self._send(200, {"ok": True})
            threading.Thread(target=self.server.shutdown, daemon=True).start()
            return
        self._send(404, {"error": "not found"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8417)
    a = ap.parse_args()
    print(f"ScanSpeak on http://localhost:{a.port}  (tumor model: {BACKEND.tumor_model is not None})")
    ThreadingHTTPServer(("127.0.0.1", a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
