"""ONNX Runtime anomaly inspector. This is the class the ROS 2 node wraps: no torch, no anomalib."""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import onnxruntime as ort

Device = Literal["cuda", "cpu"]
DEFAULT_INPUT_HW: tuple[int, int] = (256, 256)


@dataclass(frozen=True)
class InspectionResult:
    score: float
    is_anomalous: bool
    anomaly_map: np.ndarray | None
    inference_ms: float
    total_ms: float


def _preload_cuda_libraries() -> None:
    """Load CUDA/cuDNN from the nvidia-* pip wheels when available (onnxruntime >= 1.21)."""
    preload = getattr(ort, "preload_dlls", None)
    if callable(preload):
        preload()


class OnnxInspector:
    def __init__(
        self,
        model_path: Path,
        device: Device = "cuda",
        threshold: float = 0.5,
        intra_op_threads: int = 0,
    ) -> None:
        if not model_path.is_file():
            raise FileNotFoundError(f"ONNX model not found: {model_path}")

        self.threshold = threshold
        self.device: Device = device

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.intra_op_num_threads = intra_op_threads

        if device == "cuda":
            _preload_cuda_libraries()
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

        self._session = ort.InferenceSession(str(model_path), sess_options=options, providers=providers)
        active = self._session.get_providers()
        if device == "cuda" and "CUDAExecutionProvider" not in active:
            raise RuntimeError(
                f"CUDAExecutionProvider not active (got {active}). "
                "Check onnxruntime-gpu and the CUDA 12 / cuDNN 9 libraries."
            )

        model_input = self._session.get_inputs()[0]
        self._input_name: str = model_input.name
        self.input_hw: tuple[int, int] = self._resolve_hw(model_input.shape)
        self._output_names: list[str] = [o.name for o in self._session.get_outputs()]

    @staticmethod
    def _resolve_hw(shape: Sequence[int | str | None]) -> tuple[int, int]:
        if len(shape) != 4:
            raise ValueError(f"Expected NCHW input, got shape {shape}")
        h, w = shape[2], shape[3]
        if isinstance(h, int) and isinstance(w, int):
            return h, w
        return DEFAULT_INPUT_HW

    @property
    def providers(self) -> list[str]:
        return list(self._session.get_providers())

    @property
    def output_names(self) -> list[str]:
        return list(self._output_names)

    def preprocess(self, image_bgr: np.ndarray) -> np.ndarray:
        if image_bgr.ndim == 2:
            image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
        if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            raise ValueError(f"Expected HxWx3 BGR image, got shape {image_bgr.shape}")
        h, w = self.input_hw
        resized = cv2.resize(image_bgr, (w, h), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = rgb.astype(np.float32) / 255.0
        return np.ascontiguousarray(tensor.transpose(2, 0, 1)[np.newaxis])

    def run(self, tensor: np.ndarray) -> dict[str, np.ndarray]:
        outputs = self._session.run(None, {self._input_name: tensor})
        return dict(zip(self._output_names, outputs, strict=True))

    def _parse(self, outputs: dict[str, np.ndarray]) -> tuple[float, np.ndarray | None]:
        score_arr = outputs.get("pred_score")
        map_arr = outputs.get("anomaly_map")
        if score_arr is None or map_arr is None:
            for arr in outputs.values():
                if arr.dtype.kind != "f":
                    continue
                if score_arr is None and arr.size == 1:
                    score_arr = arr
                elif map_arr is None and arr.ndim >= 3:
                    map_arr = arr
        if score_arr is None:
            raise RuntimeError(f"No anomaly score found among outputs {self._output_names}")
        anomaly_map = np.squeeze(map_arr).astype(np.float32) if map_arr is not None else None
        return float(np.asarray(score_arr).reshape(-1)[0]), anomaly_map

    def infer(self, image_bgr: np.ndarray) -> InspectionResult:
        t0 = time.perf_counter()
        tensor = self.preprocess(image_bgr)
        t1 = time.perf_counter()
        outputs = self.run(tensor)
        t2 = time.perf_counter()
        score, anomaly_map = self._parse(outputs)
        t3 = time.perf_counter()
        return InspectionResult(
            score=score,
            is_anomalous=score >= self.threshold,
            anomaly_map=anomaly_map,
            inference_ms=(t2 - t1) * 1e3,
            total_ms=(t3 - t0) * 1e3,
        )

    def warmup(self, iterations: int = 10) -> None:
        h, w = self.input_hw
        dummy = np.zeros((1, 3, h, w), dtype=np.float32)
        for _ in range(iterations):
            self.run(dummy)
