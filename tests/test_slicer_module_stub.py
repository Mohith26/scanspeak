"""Exercise the Slicer module's op translation with stubbed slicer/qt/vtk.

This can't prove rendering is right (that needs real Slicer), but it does catch
Python errors and wrong call shapes in every op path, without Slicer installed.
"""
import os, sys, types
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
for name in ["slicer", "qt", "vtk", "slicer.ScriptedLoadableModule"]:
    sys.modules[name] = mock.MagicMock()
class _Base:  # real base classes so subclassing works
    def __init__(self, *a, **k): pass
sys.modules["slicer.ScriptedLoadableModule"].ScriptedLoadableModuleLogic = _Base
sys.modules["slicer.ScriptedLoadableModule"].ScriptedLoadableModule = _Base
sys.modules["slicer.ScriptedLoadableModule"].ScriptedLoadableModuleWidget = _Base
sys.path.insert(0, os.path.join(ROOT, "slicer", "ScanSpeak"))
import ScanSpeak  # noqa: E402

logic = ScanSpeak.ScanSpeakLogic()
logic.backend.zooms = (0.21, 0.21, 0.21)
img = mock.MagicMock(); img.GetDimensions.return_value = (192, 192, 480)
vol = mock.MagicMock(); vol.GetImageData.return_value = img
ScanSpeak.slicer.util.loadVolume.return_value = vol
seg = mock.MagicMock(); seg.GetSegmentation.return_value.GetSegmentIDs.side_effect = [[], ["s1"]] * 10
ScanSpeak.slicer.mrmlScene.AddNewNodeByClass.return_value = seg
ScanSpeak.vtk.vtkMatrix4x4.return_value.MultiplyPoint.return_value = (1.0, 2.0, 3.0, 1.0)

ops = [
    {"op": "load", "url": "/cache/M37_day8_ct.nii.gz", "min": -200, "max": 300},
    {"op": "window", "min": -100, "max": 1000},
    {"op": "overlay", "structure": "tumor", "url": "/cache/M37_day8_tumor.nii.gz", "color": "red", "opacity": 0.5},
    {"op": "opacity", "structure": "tumor", "opacity": 0.2},
    {"op": "layout", "layout": "three_d"},
    {"op": "focus", "frac": [0.5, 0.5, 0.5]},
    {"op": "reset", "min": -200, "max": 300},
    {"op": "download", "url": "/cache/measurements.csv"},
]
for op in ops:
    logic.apply(op)
    print("ok", op["op"])
assert vol.SetSpacing.called and ScanSpeak.slicer.vtkMRMLSliceNode.JumpAllSlices.called
print("all op paths executed")
