"""Baseline PatchCore training and evaluation on a single MVTec AD category."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import Patchcore
from lightning.pytorch import seed_everything


@dataclass(frozen=True)
class BaselineConfig:
    dataset_root: Path
    category: str
    output_dir: Path
    backbone: str
    coreset_ratio: float
    batch_size: int
    num_workers: int
    seed: int


def parse_args() -> BaselineConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path.home() / "datasets" / "MVTecAD")
    parser.add_argument("--category", type=str, default="bottle")
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--backbone", type=str, default="wide_resnet50_2")
    parser.add_argument("--coreset-ratio", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    return BaselineConfig(
        dataset_root=args.dataset_root.expanduser().resolve(),
        category=args.category,
        output_dir=args.output_dir.resolve(),
        backbone=args.backbone,
        coreset_ratio=args.coreset_ratio,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )


def to_float_dict(raw: dict[str, Any]) -> dict[str, float]:
    return {k: float(v.item() if isinstance(v, torch.Tensor) else v) for k, v in raw.items()}


def run(cfg: BaselineConfig) -> dict[str, Any]:
    seed_everything(cfg.seed, workers=True)
    torch.set_float32_matmul_precision("high")

    datamodule = MVTecAD(
        root=cfg.dataset_root,
        category=cfg.category,
        train_batch_size=cfg.batch_size,
        eval_batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
    )
    model = Patchcore(
        backbone=cfg.backbone,
        layers=("layer2", "layer3"),
        pre_trained=True,
        coreset_sampling_ratio=cfg.coreset_ratio,
    )
    engine = Engine(default_root_dir=cfg.output_dir, accelerator="auto", devices=1)

    t0 = time.perf_counter()
    engine.fit(datamodule=datamodule, model=model)
    fit_seconds = time.perf_counter() - t0

    t0 = time.perf_counter()
    test_results = engine.test(datamodule=datamodule, model=model)
    test_seconds = time.perf_counter() - t0

    report: dict[str, Any] = {
        "model": "Patchcore",
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(cfg).items()},
        "metrics": to_float_dict(test_results[0]),
        "timing_s": {"fit": round(fit_seconds, 2), "test": round(test_seconds, 2)},
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    out_file = cfg.output_dir / f"baseline_patchcore_{cfg.category}.json"
    out_file.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"\nReport saved to {out_file}")
    return report


if __name__ == "__main__":
    run(parse_args())
