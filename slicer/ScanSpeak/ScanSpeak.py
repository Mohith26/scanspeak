"""ScanSpeak as a 3D Slicer scripted module.

Same agent, same tool schema, same measurement code as the browser app; only
the "view ops" are applied to Slicer's MRML scene instead of NiiVue.

Install: Slicer > Edit > Application Settings > Modules > Additional module
paths > add this folder (slicer/ScanSpeak), restart, then find ScanSpeak under
the "Examples" category. Tumor segmentation needs torch + monai inside Slicer:
the module offers to pip-install them on first use.
"""

import os
import sys

import qt
import slicer
import vtk
from slicer.ScriptedLoadableModule import (ScriptedLoadableModule, ScriptedLoadableModuleLogic,
                                           ScriptedLoadableModuleWidget)

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

LAYOUTS = {
    "axial": slicer.vtkMRMLLayoutNode.SlicerLayoutOneUpRedSliceView,
    "sagittal": slicer.vtkMRMLLayoutNode.SlicerLayoutOneUpYellowSliceView,
    "coronal": slicer.vtkMRMLLayoutNode.SlicerLayoutOneUpGreenSliceView,
    "three_d": slicer.vtkMRMLLayoutNode.SlicerLayoutOneUp3DView,
    "four_up": slicer.vtkMRMLLayoutNode.SlicerLayoutFourUpView,
}
RGB = {"red": (0.9, 0.25, 0.25), "blue": (0.3, 0.5, 0.95), "warm": (0.95, 0.85, 0.6), "green": (0.3, 0.85, 0.45)}


class ScanSpeak(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        parent.title = "ScanSpeak"
        parent.categories = ["Examples"]
        parent.contributors = ["Mohith Gajjela"]
        parent.helpText = ("Type plain-English requests about a preclinical micro-CT scan. A local "
                           "LLM (Ollama) turns them into tool calls; measurements come from the tools.")
        parent.acknowledgementText = "Data: TumSeg database (Jensen et al., Sci Data 2024, CC-BY)."


class ScanSpeakLogic(ScriptedLoadableModuleLogic):
    """Runs the shared VolumeBackend and mirrors its view ops into the MRML scene."""

    def __init__(self):
        ScriptedLoadableModuleLogic.__init__(self)
        from scanspeak.backends.volume_backend import VolumeBackend
        self.backend = VolumeBackend()
        self.volume_node = None
        self.seg_node = None

    # ------------------------------------------------------------ ops
    def _cache_path(self, url):
        from scanspeak.backends.volume_backend import CACHE
        return os.path.realpath(os.path.join(CACHE, url.split("/cache/", 1)[1]))

    def _fix_geometry(self, node):
        # TumSeg sform is identity; real spacing lives in pixdim (0.21 mm).
        node.SetSpacing(*[float(z) for z in self.backend.zooms])
        node.SetOrigin(0.0, 0.0, 0.0)
        node.SetIJKToRASDirections(1, 0, 0, 0, 1, 0, 0, 0, 1)

    def apply(self, op):
        kind = op["op"]
        if kind == "load":
            if self.seg_node:
                slicer.mrmlScene.RemoveNode(self.seg_node)
                self.seg_node = None
            if self.volume_node:
                slicer.mrmlScene.RemoveNode(self.volume_node)
            self.volume_node = slicer.util.loadVolume(self._cache_path(op["url"]))
            self._fix_geometry(self.volume_node)
            self._window(op["min"], op["max"])
            slicer.util.setSliceViewerLayers(background=self.volume_node, fit=True)
        elif kind == "window":
            self._window(op["min"], op["max"])
        elif kind == "overlay":
            self._overlay(op)
        elif kind == "opacity":
            self._opacity(op["structure"], op["opacity"])
        elif kind == "layout":
            slicer.app.layoutManager().setLayout(LAYOUTS[op["layout"]])
            if op["layout"] == "three_d":
                self._center_3d()
        elif kind == "focus":
            self._focus(op["frac"])
        elif kind == "reset":
            slicer.app.layoutManager().setLayout(LAYOUTS["four_up"])
            self._window(op["min"], op["max"])
            slicer.util.resetSliceViews()
            self._center_3d()
        elif kind == "download":
            slicer.util.infoDisplay("Saved: " + self._cache_path(op["url"]))

    def _window(self, lo, hi):
        if self.volume_node:
            d = self.volume_node.GetDisplayNode()
            d.SetAutoWindowLevel(False)
            d.SetWindowLevelMinMax(lo, hi)

    def _ensure_seg(self):
        if self.seg_node is None:
            self.seg_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", "ScanSpeak")
            self.seg_node.CreateDefaultDisplayNodes()
            self.seg_node.SetReferenceImageGeometryParameterFromVolumeNode(self.volume_node)
        return self.seg_node

    def _overlay(self, op):
        seg = self._ensure_seg()
        name = op["structure"]
        old = seg.GetSegmentation().GetSegmentIdBySegmentName(name)
        if old:
            seg.GetSegmentation().RemoveSegment(old)
        label = slicer.util.loadLabelVolume(self._cache_path(op["url"]))
        self._fix_geometry(label)
        before = set(seg.GetSegmentation().GetSegmentIDs())
        slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(label, seg)
        slicer.mrmlScene.RemoveNode(label)
        for sid in set(seg.GetSegmentation().GetSegmentIDs()) - before:
            s = seg.GetSegmentation().GetSegment(sid)
            s.SetName(name)
            s.SetColor(*RGB.get(op["color"], (1, 0, 0)))
            seg.GetDisplayNode().SetSegmentOpacity2DFill(sid, op["opacity"])
        seg.CreateClosedSurfaceRepresentation()

    def _segment_id(self, structure):
        if self.seg_node is None:
            return None
        return self.seg_node.GetSegmentation().GetSegmentIdBySegmentName(structure) or None

    def _opacity(self, structure, value):
        sid = self._segment_id(structure)
        if sid:
            d = self.seg_node.GetDisplayNode()
            d.SetSegmentVisibility(sid, value > 0)
            d.SetSegmentOpacity2DFill(sid, value)
            d.SetSegmentOpacity3D(sid, max(value, 0.05))

    def _focus(self, frac):
        dims = self.volume_node.GetImageData().GetDimensions()
        ijk = [f * d - 0.5 for f, d in zip(frac, dims)] + [1.0]
        m = vtk.vtkMatrix4x4()
        self.volume_node.GetIJKToRASMatrix(m)
        ras = m.MultiplyPoint(ijk)[:3]
        slicer.vtkMRMLSliceNode.JumpAllSlices(slicer.mrmlScene, *ras,
                                              slicer.vtkMRMLSliceNode.CenteredJumpSlice)

    def _center_3d(self):
        view = slicer.app.layoutManager().threeDWidget(0).threeDView()
        view.resetFocalPoint()
        view.resetCamera()

    # ------------------------------------------------------------ chat
    def run(self, text, model, mode):
        from scanspeak.agent import plan
        p = plan(text, self.backend.state_text(), model, mode)
        if p["problems"]:
            return p, [], "Couldn't form a valid action: " + "; ".join(p["problems"][:2])
        results = self.backend.execute(p["calls"])
        for r in results:
            for op in r.get("ops", []):
                self.apply(op)
        reply = " ".join(r["text"] for r in results) if p["calls"] else (p["reply"] or "Not something I can do.")
        return p, results, reply


class ScanSpeakWidget(ScriptedLoadableModuleWidget):
    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)
        self.logic = ScanSpeakLogic()
        form = qt.QFormLayout()
        self.model = qt.QComboBox()
        try:
            from scanspeak.agent import available_models
            names = [m for m in available_models() if "embed" not in m]
        except Exception:
            names = []
        self.model.addItems(names or ["qwen2.5:3b"])
        self.mode = qt.QComboBox()
        self.mode.addItems(["strict", "native", "schema", "prompt"])
        form.addRow("Model:", self.model)
        form.addRow("Decoding:", self.mode)
        self.layout.addLayout(form)

        self.log = qt.QTextBrowser()
        self.log.setMinimumHeight(320)
        self.layout.addWidget(self.log)

        row = qt.QHBoxLayout()
        self.input = qt.QLineEdit()
        self.input.setPlaceholderText("e.g. open M37 at day 8 and how big is the tumor?")
        self.input.returnPressed.connect(self.onRun)
        run = qt.QPushButton("Run")
        run.clicked.connect(self.onRun)
        row.addWidget(self.input)
        row.addWidget(run)
        self.layout.addLayout(row)
        self.layout.addStretch(1)

        if self.logic.backend.tumor_model is None:
            self.log.append("<i>No tumor model set. Export SCANSPEAK_TUMOR_MODEL=/path/model.pt before "
                            "launching Slicer, and pip_install('torch monai') in the Python console.</i>")

    def onRun(self):
        text = self.input.text.strip()
        if not text:
            return
        self.input.clear()
        self.log.append(f"<b>you:</b> {text}")
        slicer.app.processEvents()
        with slicer.util.tryWithErrorDisplay("ScanSpeak failed", waitCursor=True):
            p, results, reply = self.logic.run(text, self.model.currentText, self.mode.currentText)
            calls = ", ".join(f"{c['name']}({', '.join(f'{k}={v!r}' for k, v in c['arguments'].items())})"
                              for c in p["calls"])
            if calls:
                self.log.append(f"<span style='color:#2a9d8f;font-family:monospace'>{calls}</span>")
            self.log.append(f"{reply} <span style='color:gray'>(LLM {p['latency_s']:.1f}s)</span>")
