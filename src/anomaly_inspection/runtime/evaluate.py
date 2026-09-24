"""Evaluate an exported ONNX model on an MVTec AD category: image-level accuracy and latency."""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, get_args

import cv2
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

from anomaly_inspection.runtime.onnx_inspector import Device, OnnxInspector

LOGGER = logging.getLogger("anomaly_inspection.evaluate")
IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".bmp"})


@dataclass(frozen=True)
class Sample:
    path: Path
    label: int  # 0 = good, 1 = anomalous


def list_test_samples(dataset_root: Path, category: str) -> list[Sample]:
    test_dir = dataset_root / category / "test"
    if not test_dir.is_dir():
        raise FileNotFoundError(f"Test split not found: {test_dir}")
    samples = [
        Sample(path=p, label=0 if p.parent.name == "good" else 1)
        for p in sorted(test_dir.rglob("*"))
        if p.suffix.lower() in IMAGE_SUFFIXES
    ]
    if not samples:
        raise FileNotFoundError(f"No test images under {test_dir}")
    return samples


def latency_stats(values_ms: Sequence[float]) -> dict[str, float]:
    arr = np.asarray(values_ms, dtype=np.float64)
    mean = float(arr.mean())
    return {
        "mean": round(mean, 2),
        "p50": round(float(np.percentile(arr, 50)), 2),
        "p95": round(float(np.percentile(arr, 95)), 2),
        "p99": round(float(np.percentile(arr, 99)), 2),
        "fps": round(1000.0 / mean, 1),
    }


def evaluate(
    model_path: Path,
    dataset_root: Path,
    category: str,
    device: Device,
    threshold: float,
    warmup: int,
) -> dict[str, Any]:
    inspector = OnnxInspector(model_path, device=device, threshold=threshold)
    LOGGER.info("[%s] providers=%s input=%s outputs=%s", device, inspector.providers, inspector.input_hw, inspector.output_names)
    inspector.warmup(warmup)

    samples = list_test_samples(dataset_root, category)
    scores: list[float] = []
    inference_ms: list[float] = []
    total_ms: list[float] = []

    for sample in samples:
        image = cv2.imread(str(sample.path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Could not read image {sample.path}")
        result = inspector.infer(image)
        scores.append(result.score)
        inference_ms.append(result.inference_ms)
        total_ms.append(result.total_ms)

    labels = np.array([s.label for s in samples])
    preds = (np.asarray(scores) >= threshold).astype(int)

    return {
        "model_path": str(model_path),
        "category": category,
        "device": device,
        "providers": inspector.providers,
        "input_hw": list(inspector.input_hw),
        "n_images": len(samples),
        "threshold": threshold,
        "metrics": {
            "image_AUROC": round(float(roc_auc_score(labels, scores)), 4),
            "image_F1_at_threshold": round(float(f1_score(labels, preds)), 4),
            "accuracy_at_threshold": round(float((preds == labels).mean()), 4),
        },
        "latency_ms": {
            "inference": latency_stats(inference_ms),
            "end_to_end": latency_stats(total_ms),
        },
    }


def print_table(reports: Sequence[dict[str, Any]]) -> None:
    header = "| device | AUROC | F1@thr | inf p50 [ms] | inf p95 [ms] | e2e p50 [ms] | FPS (e2e) |"
    print("\n" + header)
    print("|" + "---|" * 7)
    for r in reports:
        m, inf, e2e = r["metrics"], r["latency_ms"]["inference"], r["latency_ms"]["end_to_end"]
        print(
            f"| {r['device']} | {m['image_AUROC']:.3f} | {m['image_F1_at_threshold']:.3f} | "
            f"{inf['p50']:.2f} | {inf['p95']:.2f} | {e2e['p50']:.2f} | {e2e['fps']:.1f} |"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--category", required=True)
    parser.add_argument("--dataset-root", type=Path, default=Path.home() / "datasets" / "MVTecAD")
    parser.add_argument("--devices", nargs="+", choices=get_args(Device), default=["cuda", "cpu"])
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, default=Path("results/onnx"))
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()
    model_path: Path = args.model_path.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    reports: list[dict[str, Any]] = []
    for device in args.devices:
        report = evaluate(
            model_path=model_path,
            dataset_root=args.dataset_root.expanduser().resolve(),
            category=args.category,
            device=device,
            threshold=args.threshold,
            warmup=args.warmup,
        )
        out = args.output_dir / f"{model_path.stem}_{device}.json"
        out.write_text(json.dumps(report, indent=2))
        LOGGER.info("Report saved to %s", out)
        reports.append(report)

    print_table(reports)


if __name__ == "__main__":
    main()
