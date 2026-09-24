"""Torch-free inference runtime (onnxruntime + numpy + opencv only)."""

from anomaly_inspection.runtime.onnx_inspector import InspectionResult, OnnxInspector

__all__ = ["InspectionResult", "OnnxInspector"]
