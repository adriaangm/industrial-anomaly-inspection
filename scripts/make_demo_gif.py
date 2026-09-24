"""Render the README demo GIF: MVTec test images + anomaly heatmap + verdict, via the torch-free runtime."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from anomaly_inspection.runtime import OnnxInspector

IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".bmp"})
PANEL_SIZE = 320
BANNER_H = 44


def render_frame(image_bgr: np.ndarray, anomaly_map: np.ndarray, score: float, is_anomalous: bool, threshold: float) -> np.ndarray:
    image = cv2.resize(image_bgr, (PANEL_SIZE, PANEL_SIZE), interpolation=cv2.INTER_AREA)
    heat_u8 = (np.clip(anomaly_map, 0.0, 1.0) * 255).astype(np.uint8)
    heat = cv2.applyColorMap(cv2.resize(heat_u8, (PANEL_SIZE, PANEL_SIZE)), cv2.COLORMAP_JET)
    blended = cv2.addWeighted(image, 0.55, heat, 0.45, 0)

    frame = np.zeros((PANEL_SIZE + BANNER_H, PANEL_SIZE * 2, 3), dtype=np.uint8)
    frame[BANNER_H:, :PANEL_SIZE] = image
    frame[BANNER_H:, PANEL_SIZE:] = blended

    color = (40, 40, 230) if is_anomalous else (60, 200, 60)
    label = f"{'DEFECT' if is_anomalous else 'OK'}   score {score:.2f}  (thr {threshold:.2f})"
    cv2.putText(frame, label, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)
    return frame


def pick_samples(test_dir: Path, n_good: int, n_defect: int, seed: int) -> list[Path]:
    paths = sorted(p for p in test_dir.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    good = [p for p in paths if p.parent.name == "good"]
    defect = [p for p in paths if p.parent.name != "good"]
    rng = random.Random(seed)
    chosen = rng.sample(good, min(n_good, len(good))) + rng.sample(defect, min(n_defect, len(defect)))
    rng.shuffle(chosen)
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=Path("models/patchcore_metal_nut.onnx"))
    parser.add_argument("--category", default="metal_nut")
    parser.add_argument("--dataset-root", type=Path, default=Path.home() / "datasets" / "MVTecAD")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--n-good", type=int, default=4)
    parser.add_argument("--n-defect", type=int, default=8)
    parser.add_argument("--frame-ms", type=int, default=900)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=Path("docs/demo.gif"))
    args = parser.parse_args()

    inspector = OnnxInspector(args.model_path.resolve(), device=args.device)
    inspector.warmup(5)
    test_dir = args.dataset_root.expanduser() / args.category / "test"

    frames: list[Image.Image] = []
    for path in pick_samples(test_dir, args.n_good, args.n_defect, args.seed):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Could not read {path}")
        result = inspector.infer(image)
        if result.anomaly_map is None:
            raise RuntimeError("Model does not output an anomaly map")
        frame = render_frame(image, result.anomaly_map, result.score, result.is_anomalous, inspector.threshold)
        frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        print(f"{path.parent.name:>14}/{path.name}  score={result.score:.3f}  anomalous={result.is_anomalous}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(args.output, save_all=True, append_images=frames[1:], duration=args.frame_ms, loop=0, optimize=True)
    print(f"GIF saved to {args.output} ({args.output.stat().st_size / 2**20:.1f} MB, {len(frames)} frames)")


if __name__ == "__main__":
    main()
