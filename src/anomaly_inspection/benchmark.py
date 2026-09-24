"""Multi-model, multi-category anomaly detection benchmark on MVTec AD."""

from __future__ import annotations

import argparse
import gc
import json
import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import EfficientAd, Padim, Patchcore
from lightning.pytorch import LightningModule, seed_everything

from anomaly_inspection.config import BenchmarkConfig, ModelSpec, load_config

LOGGER = logging.getLogger("anomaly_inspection.benchmark")

MODEL_REGISTRY: dict[str, Callable[..., LightningModule]] = {
    "patchcore": Patchcore,
    "padim": Padim,
    "efficient_ad": EfficientAd,
}

SUMMARY_SECTIONS: tuple[tuple[str, str, str], ...] = (
    ("image_AUROC", "Image-level AUROC", "{:.3f}"),
    ("pixel_AUROC", "Pixel-level AUROC", "{:.3f}"),
    ("image_F1Score", "Image-level F1", "{:.3f}"),
    ("fit_s", "Training time [s]", "{:.1f}"),
    ("peak_vram_mb", "Peak VRAM [MB]", "{:.0f}"),
)


@dataclass
class RunResult:
    model: str
    category: str
    status: str
    metrics: dict[str, float] = field(default_factory=dict)
    fit_s: float | None = None
    test_s: float | None = None
    peak_vram_mb: float | None = None
    error: str | None = None

    @classmethod
    def from_json(cls, path: Path) -> RunResult:
        return cls(**json.loads(path.read_text()))

    def to_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2))


def run_path(cfg: BenchmarkConfig, model_name: str, category: str) -> Path:
    return cfg.runs_dir / f"{model_name}__{category}.json"


def build_model(spec: ModelSpec) -> LightningModule:
    try:
        factory = MODEL_REGISTRY[spec.name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown model '{spec.name}'. Available: {sorted(MODEL_REGISTRY)}"
        ) from exc
    return factory(**spec.params)


def _to_float(value: Any) -> float:
    return float(value.item() if isinstance(value, torch.Tensor) else value)


def release_gpu() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_single(cfg: BenchmarkConfig, spec: ModelSpec, category: str) -> RunResult:
    seed_everything(cfg.seed, workers=True)
    datamodule = MVTecAD(
        root=cfg.dataset_root,
        category=category,
        train_batch_size=spec.train_batch_size or cfg.train_batch_size,
        eval_batch_size=cfg.eval_batch_size,
        num_workers=cfg.num_workers,
    )
    model = build_model(spec)
    engine = Engine(
        default_root_dir=cfg.anomalib_dir,
        accelerator="auto",
        devices=1,
        **spec.trainer,
    )

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    engine.fit(datamodule=datamodule, model=model)
    fit_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    test_results = engine.test(datamodule=datamodule, model=model)
    test_s = time.perf_counter() - t0

    peak_mb = torch.cuda.max_memory_allocated() / 2**20 if use_cuda else None

    return RunResult(
        model=spec.name,
        category=category,
        status="ok",
        metrics={k: _to_float(v) for k, v in test_results[0].items()},
        fit_s=round(fit_s, 2),
        test_s=round(test_s, 2),
        peak_vram_mb=round(peak_mb, 1) if peak_mb is not None else None,
    )


def run_benchmark(
    cfg: BenchmarkConfig,
    models: Sequence[ModelSpec],
    categories: Sequence[str],
    force: bool,
) -> list[RunResult]:
    results: list[RunResult] = []
    total = len(models) * len(categories)
    index = 0

    for spec in models:
        for category in categories:
            index += 1
            path = run_path(cfg, spec.name, category)

            if path.exists() and not force:
                cached = RunResult.from_json(path)
                if cached.status == "ok":
                    LOGGER.info("[%d/%d] %s / %s: cached, skipping", index, total, spec.name, category)
                    results.append(cached)
                    continue

            LOGGER.info("[%d/%d] %s / %s: running", index, total, spec.name, category)
            try:
                result = run_single(cfg, spec, category)
            except Exception as exc:  # noqa: BLE001 - a single failed run must not abort the benchmark
                LOGGER.exception("%s / %s failed", spec.name, category)
                result = RunResult(
                    model=spec.name,
                    category=category,
                    status="failed",
                    error=f"{type(exc).__name__}: {exc}",
                )
            finally:
                release_gpu()

            result.to_json(path)
            results.append(result)
            LOGGER.info(
                "[%d/%d] %s / %s: %s %s",
                index, total, spec.name, category, result.status,
                {k: round(v, 4) for k, v in result.metrics.items()},
            )

    return results


def results_to_frame(results: Sequence[RunResult]) -> pd.DataFrame:
    rows = [
        {
            "model": r.model,
            "category": r.category,
            "status": r.status,
            **r.metrics,
            "fit_s": r.fit_s,
            "test_s": r.test_s,
            "peak_vram_mb": r.peak_vram_mb,
        }
        for r in results
    ]
    return pd.DataFrame(rows)


def pivot_metric(
    df: pd.DataFrame,
    metric: str,
    model_order: Sequence[str],
    category_order: Sequence[str],
) -> pd.DataFrame | None:
    ok = df[df["status"] == "ok"]
    if ok.empty or metric not in ok.columns:
        return None
    table = ok.pivot_table(index="category", columns="model", values=metric, aggfunc="first")
    table = table.reindex(
        index=[c for c in category_order if c in table.index],
        columns=[m for m in model_order if m in table.columns],
    )
    table.loc["mean"] = table.mean(axis=0)
    return table


def table_to_markdown(table: pd.DataFrame, fmt: str) -> str:
    header = ["category", *(str(c) for c in table.columns)]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for idx, row in table.iterrows():
        label = f"**{idx}**" if idx == "mean" else str(idx)
        cells = [fmt.format(v) if pd.notna(v) else "—" for v in row.to_numpy()]
        lines.append("| " + " | ".join([label, *cells]) + " |")
    return "\n".join(lines)


def write_summary(
    cfg: BenchmarkConfig,
    results: Sequence[RunResult],
    model_order: Sequence[str],
    category_order: Sequence[str],
) -> Path:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    df = results_to_frame(results)
    df.to_csv(cfg.output_dir / "summary.csv", index=False)

    sections = ["# MVTec AD benchmark", ""]
    for metric, title, fmt in SUMMARY_SECTIONS:
        table = pivot_metric(df, metric, model_order, category_order)
        if table is None:
            continue
        sections += [f"## {title}", "", table_to_markdown(table, fmt), ""]

    failed = [r for r in results if r.status != "ok"]
    if failed:
        sections += ["## Failed runs", ""]
        sections += [f"- `{r.model}` / `{r.category}`: {r.error}" for r in failed]
        sections.append("")

    out = cfg.output_dir / "summary.md"
    out.write_text("\n".join(sections))
    print("\n".join(sections))
    LOGGER.info("Summary written to %s", out)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/benchmark.yaml"))
    parser.add_argument("--models", nargs="+", default=None, help="Subset of model names from the config")
    parser.add_argument("--categories", nargs="+", default=None, help="Subset of categories from the config")
    parser.add_argument("--force", action="store_true", help="Re-run even if a cached result exists")
    parser.add_argument("--summary-only", action="store_true", help="Only rebuild the summary from cached runs")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()
    cfg = load_config(args.config)
    torch.set_float32_matmul_precision("high")

    models = tuple(m for m in cfg.models if args.models is None or m.name in args.models)
    categories = tuple(c for c in cfg.categories if args.categories is None or c in args.categories)
    if not models or not categories:
        raise SystemExit("No runs selected: check --models / --categories against the config.")

    if not args.summary_only:
        run_benchmark(cfg, models, categories, force=args.force)

    # The summary always covers every cached run in the config, not only this invocation's subset.
    all_results = [
        RunResult.from_json(p)
        for m in cfg.models
        for c in cfg.categories
        if (p := run_path(cfg, m.name, c)).exists()
    ]
    write_summary(cfg, all_results, [m.name for m in cfg.models], list(cfg.categories))


if __name__ == "__main__":
    main()
