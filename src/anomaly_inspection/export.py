"""Export a benchmarked Anomalib model to a single self-contained ONNX file."""

from __future__ import annotations

import argparse
import json
import logging
import tempfile
from pathlib import Path
from typing import Any, cast

import onnx
from lightning.pytorch import LightningModule

from anomaly_inspection.benchmark import MODEL_REGISTRY, RunResult, run_path
from anomaly_inspection.config import load_config

LOGGER = logging.getLogger("anomaly_inspection.export")


def find_checkpoint(anomalib_dir: Path, class_name: str, category: str) -> Path:
    root = anomalib_dir / class_name
    candidates = {p.resolve() for p in root.rglob("model.ckpt") if category in p.parts}
    if not candidates:
        raise FileNotFoundError(
            f"No checkpoint for {class_name}/{category} under {root}. Run aii-benchmark first."
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _describe_values(values: Any) -> list[dict[str, Any]]:
    described: list[dict[str, Any]] = []
    for value in values:
        dims = [
            d.dim_value if d.HasField("dim_value") else (d.dim_param or "?")
            for d in value.type.tensor_type.shape.dim
        ]
        described.append({"name": value.name, "shape": dims})
    return described


def export_model(
    model_name: str,
    category: str,
    config_path: Path,
    output_dir: Path,
    input_size: tuple[int, int],
) -> Path:
    cfg = load_config(config_path)
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{model_name}'. Available: {sorted(MODEL_REGISTRY)}")
    model_cls = cast(type[LightningModule], MODEL_REGISTRY[model_name])

    ckpt = find_checkpoint(cfg.anomalib_dir, model_cls.__name__, category)
    LOGGER.info("Loading checkpoint %s", ckpt)
    # weights_only=False: our own checkpoints store anomalib enums in hparams (trusted source).
    model = model_cls.load_from_checkpoint(str(ckpt), map_location="cpu", weights_only=False)
    model.eval()

    output_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{model_name}_{category}"
    dest = output_dir / f"{tag}.onnx"

    with tempfile.TemporaryDirectory() as tmp:
        raw_path = Path(model.to_onnx(export_root=Path(tmp), input_size=input_size))
        LOGGER.info("Raw export at %s", raw_path)
        proto = onnx.load(str(raw_path), load_external_data=True)
        onnx.checker.check_model(proto)
        onnx.save_model(proto, str(dest), save_as_external_data=False)

    inputs = _describe_values(proto.graph.input)
    outputs = _describe_values(proto.graph.output)

    benchmark_file = run_path(cfg, model_name, category)
    benchmark_metrics = (
        RunResult.from_json(benchmark_file).metrics if benchmark_file.exists() else {}
    )

    metadata = {
        "model": model_name,
        "category": category,
        "input_size": list(input_size),
        "checkpoint": str(ckpt),
        "onnx_file": str(dest),
        "size_mb": round(dest.stat().st_size / 2**20, 1),
        "inputs": inputs,
        "outputs": outputs,
        "benchmark_metrics": benchmark_metrics,
    }
    dest.with_suffix(".json").write_text(json.dumps(metadata, indent=2))

    LOGGER.info("Exported %s (%.1f MB)", dest, metadata["size_mb"])
    LOGGER.info("Inputs:  %s", inputs)
    LOGGER.info("Outputs: %s", outputs)
    return dest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/benchmark.yaml"))
    parser.add_argument("--model", required=True, choices=sorted(MODEL_REGISTRY))
    parser.add_argument("--category", required=True)
    parser.add_argument("--input-size", type=int, nargs=2, default=(256, 256), metavar=("H", "W"))
    parser.add_argument("--output-dir", type=Path, default=Path("models"))
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()
    export_model(
        model_name=args.model,
        category=args.category,
        config_path=args.config,
        output_dir=args.output_dir.resolve(),
        input_size=(args.input_size[0], args.input_size[1]),
    )


if __name__ == "__main__":
    main()
