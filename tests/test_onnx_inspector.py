"""Smoke tests for the torch-free ONNX runtime, using a synthetic model (no GPU, no MVTec needed)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

from anomaly_inspection.runtime import OnnxInspector

INPUT_HW = (64, 64)


def _build_dummy_anomaly_model(path: Path, h: int, w: int) -> None:
    """A tiny graph shaped like an Anomalib export: NCHW in, pred_score + anomaly_map out."""
    input_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, h, w])
    score_info = helper.make_tensor_value_info("pred_score", TensorProto.FLOAT, [1])
    map_info = helper.make_tensor_value_info("anomaly_map", TensorProto.FLOAT, [1, 1, h, w])

    mean_node = helper.make_node("ReduceMean", ["input"], ["pred_score"], keepdims=0)
    identity_node = helper.make_node("ReduceMean", ["input"], ["anomaly_map_pre"], axes=[1], keepdims=1)
    reshape_node = helper.make_node("Reshape", ["anomaly_map_pre", "shape"], ["anomaly_map"])
    shape_init = helper.make_tensor("shape", TensorProto.INT64, [4], [1, 1, h, w])

    graph = helper.make_graph(
        [mean_node, identity_node, reshape_node],
        "dummy_anomaly_model",
        [input_info],
        [score_info, map_info],
        initializer=[shape_init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 10  # cap at the max IR version onnxruntime 1.30 supports (13)
    onnx.checker.check_model(model)
    onnx.save_model(model, str(path))


@pytest.fixture
def dummy_model_path(tmp_path: Path) -> Path:
    path = tmp_path / "dummy.onnx"
    _build_dummy_anomaly_model(path, *INPUT_HW)
    return path


@pytest.fixture
def inspector(dummy_model_path: Path) -> OnnxInspector:
    return OnnxInspector(dummy_model_path, device="cpu", threshold=0.5)


def test_input_hw_matches_model(inspector: OnnxInspector) -> None:
    assert inspector.input_hw == INPUT_HW


def test_preprocess_shape_and_range(inspector: OnnxInspector) -> None:
    image = np.random.randint(0, 255, size=(480, 640, 3), dtype=np.uint8)
    tensor = inspector.preprocess(image)
    assert tensor.shape == (1, 3, *INPUT_HW)
    assert tensor.dtype == np.float32
    assert 0.0 <= tensor.min() and tensor.max() <= 1.0


def test_infer_end_to_end(inspector: OnnxInspector) -> None:
    black = np.zeros((100, 100, 3), dtype=np.uint8)
    result = inspector.infer(black)
    assert result.score == pytest.approx(0.0, abs=1e-6)
    assert result.is_anomalous is False
    assert result.anomaly_map is not None
    assert result.anomaly_map.shape == INPUT_HW
    assert result.inference_ms > 0.0
    assert result.total_ms >= result.inference_ms


def test_threshold_is_respected(dummy_model_path: Path) -> None:
    strict = OnnxInspector(dummy_model_path, device="cpu", threshold=0.01)
    grey = np.full((50, 50, 3), 128, dtype=np.uint8)
    result = strict.infer(grey)
    assert result.is_anomalous is True


def test_grayscale_input_is_converted(inspector: OnnxInspector) -> None:
    gray = np.zeros((100, 100), dtype=np.uint8)
    tensor = inspector.preprocess(gray)
    assert tensor.shape == (1, 3, *INPUT_HW)


def test_rejects_wrong_ndim(inspector: OnnxInspector) -> None:
    with pytest.raises(ValueError):
        inspector.preprocess(np.zeros((10, 10, 4), dtype=np.uint8))
