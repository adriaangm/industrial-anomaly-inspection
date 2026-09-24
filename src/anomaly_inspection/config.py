"""Benchmark configuration schema and YAML loader."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

MVTEC_CATEGORIES: frozenset[str] = frozenset(
    {
        "bottle", "cable", "capsule", "carpet", "grid",
        "hazelnut", "leather", "metal_nut", "pill", "screw",
        "tile", "toothbrush", "transistor", "wood", "zipper",
    }
)


class ConfigError(ValueError):
    """Raised when the benchmark configuration is invalid."""


@dataclass(frozen=True)
class ModelSpec:
    name: str
    params: dict[str, Any] = field(default_factory=dict)
    trainer: dict[str, Any] = field(default_factory=dict)
    train_batch_size: int | None = None


@dataclass(frozen=True)
class BenchmarkConfig:
    dataset_root: Path
    output_dir: Path
    categories: tuple[str, ...]
    models: tuple[ModelSpec, ...]
    train_batch_size: int = 32
    eval_batch_size: int = 32
    num_workers: int = 4
    seed: int = 42

    @property
    def runs_dir(self) -> Path:
        return self.output_dir / "runs"

    @property
    def anomalib_dir(self) -> Path:
        return self.output_dir / "anomalib"


def _expand_user(value: Any) -> Any:
    """Recursively expand '~' in string values (Anomalib does not do it)."""
    if isinstance(value, str) and value.startswith("~"):
        return str(Path(value).expanduser())
    if isinstance(value, dict):
        return {k: _expand_user(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_user(v) for v in value]
    return value


def _parse_model(raw: Any) -> ModelSpec:
    if not isinstance(raw, dict) or "name" not in raw:
        raise ConfigError(f"Each model entry needs at least a 'name' key, got: {raw!r}")
    batch = raw.get("train_batch_size")
    return ModelSpec(
        name=str(raw["name"]),
        params=_expand_user(raw.get("params") or {}),
        trainer=dict(raw.get("trainer") or {}),
        train_batch_size=int(batch) if batch is not None else None,
    )


def load_config(path: Path) -> BenchmarkConfig:
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ConfigError(f"Top-level YAML must be a mapping: {path}")

    try:
        categories = tuple(str(c) for c in raw["categories"])
        models = tuple(_parse_model(m) for m in raw["models"])
        dataset_root = Path(raw["dataset_root"]).expanduser().resolve()
    except KeyError as exc:
        raise ConfigError(f"Missing required key in {path}: {exc}") from exc

    unknown = sorted(set(categories) - MVTEC_CATEGORIES)
    if unknown:
        raise ConfigError(f"Unknown MVTec AD categories: {unknown}")

    names = [m.name for m in models]
    duplicated = sorted({n for n in names if names.count(n) > 1})
    if duplicated:
        raise ConfigError(f"Duplicated model names: {duplicated}")

    return BenchmarkConfig(
        dataset_root=dataset_root,
        output_dir=Path(raw.get("output_dir", "results/benchmark")).expanduser().resolve(),
        categories=categories,
        models=models,
        train_batch_size=int(raw.get("train_batch_size", 32)),
        eval_batch_size=int(raw.get("eval_batch_size", 32)),
        num_workers=int(raw.get("num_workers", 4)),
        seed=int(raw.get("seed", 42)),
    )
