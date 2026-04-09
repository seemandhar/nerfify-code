#!/usr/bin/env python3
"""
Train and evaluate NeRF methods on standard datasets.

Usage:
    python eval.py --method hybrid-nerf --scenes garden
    python eval.py --method hybrid-nerf --scenes garden bicycle --gpu 0,1
    python eval.py --eval-only --method hybrid-nerf --scenes garden
    python eval.py --method hybrid-nerf --scenes lego --dataset blender
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed


# ─── Configuration ───────────────────────────────────────────────
DEFAULT_OUTPUT_ROOT = Path.home() / "research" / "NeRF2Code" / "nerfify" / "outputs"

DATASETS = {
    "mipnerf360": {
        "data_root": Path.home() / "data" / "nerfstudio" / "mipnerf360",
        "scenes": [
            "bicycle", "bonsai", "counter", "garden", "kitchen",
            "room", "stump", "flowers", "treehill",
        ],
        "dataparser": "colmap",
        "dataparser_args": ["--eval-mode", "interval"],
    },
    "blender": {
        "data_root": Path.home() / "data" / "nerf_synthetic",
        "scenes": ["chair", "drums", "ficus", "hotdog", "lego", "materials", "mic", "ship"],
        "dataparser": "blender-data",
        "dataparser_args": [],
    },
    "llff": {
        "data_root": Path.home() / "data" / "nerf_llff_data",
        "scenes": ["fern", "flower", "fortress", "horns", "leaves", "orchids", "room", "trex"],
        "dataparser": "colmap",
        "dataparser_args": ["--eval-mode", "filename"],
    },
}


def run_cmd(cmd: list[str], gpu: int = 0, timeout: int = 7200) -> tuple[int, str]:
    """Run a command with the specified GPU. Returns (returncode, stderr)."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    print(f"\n$ {' '.join(cmd)}\n")
    result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        print(result.stderr[-2000:] if result.stderr else "")
    return result.returncode, result.stderr or ""


def get_latest_config(scene: str, method: str, output_root: Path) -> Path | None:
    """Find the most recent config.yml for a scene/method combo."""
    run_dir = output_root / scene / method
    if not run_dir.exists():
        return None
    runs = sorted(run_dir.iterdir(), key=lambda p: p.name, reverse=True)
    for run in runs:
        config = run / "config.yml"
        if config.exists():
            return config
    return None


def get_latest_run_dir(scene: str, method: str, output_root: Path) -> Path | None:
    """Find the most recent run directory for a scene/method combo."""
    run_dir = output_root / scene / method
    if not run_dir.exists():
        return None
    runs = sorted(run_dir.iterdir(), key=lambda p: p.name, reverse=True)
    return runs[0] if runs else None


def train_scene(
    scene: str,
    method: str,
    gpu: int,
    max_iters: int = 3000,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    data_root: Path = None,
    dataparser: str = "colmap",
    dataparser_args: list[str] = None,
) -> tuple[bool, str]:
    """Train a single scene. Returns (success, error_output)."""
    data_path = data_root / scene
    if not data_path.exists():
        return False, f"Data not found at {data_path}"

    cmd = [
        "ns-train", method,
        "--data", str(data_path),
        "--vis", "viewer+tensorboard",
        "--max-num-iterations", str(max_iters),
        "--output-dir", str(output_root),
        "--viewer.quit-on-train-completion", "True",
        dataparser,
    ]
    if dataparser_args:
        cmd += dataparser_args

    ret, stderr = run_cmd(cmd, gpu=gpu, timeout=7200)
    if ret != 0:
        print(f"  ERROR: Training failed for {scene} (exit code {ret})")
        return False, stderr[-3000:]
    return True, ""


def eval_scene(
    scene: str,
    method: str,
    gpu: int,
    results_dir: Path,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict | None:
    """Evaluate a single scene. Returns the results dict or None."""
    config = get_latest_config(scene, method, output_root)
    if config is None:
        print(f"  WARNING: No config found for {scene}/{method}, skipping eval.")
        return None

    output_path = results_dir / f"{scene}.json"
    print(f"  Config: {config}")

    ret, _ = run_cmd(
        ["ns-eval", "--load-config", str(config), "--output-path", str(output_path)],
        gpu=gpu,
    )
    if ret != 0:
        print(f"  ERROR: Eval failed for {scene} (exit code {ret})")
        return None

    if output_path.exists():
        with open(output_path) as f:
            return json.load(f)
    return None


def read_psnr_from_tb(run_dir: Path) -> dict | None:
    """Read PSNR from TensorBoard logs using read_tb.py."""
    read_tb = Path(__file__).parent / "read_tb.py"
    try:
        result = subprocess.run(
            [sys.executable, str(read_tb), str(run_dir), "--json"],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0 and result.stdout.strip():
            return json.loads(result.stdout)
    except Exception as e:
        print(f"  WARNING: Could not read TensorBoard logs: {e}")
    return None


def print_results_table(all_results: dict[str, dict], method: str):
    """Print a formatted results table."""
    print("\n" + "=" * 70)
    print(f"  Results: {method}")
    print("=" * 70)
    print(f"  {'Scene':<12s}  {'PSNR':>8s}  {'SSIM':>8s}  {'LPIPS':>8s}  {'FPS':>8s}")
    print("-" * 70)

    metrics_agg = {"psnr": [], "ssim": [], "lpips": [], "fps": []}

    for scene in sorted(all_results.keys()):
        data = all_results.get(scene)
        if data is None:
            print(f"  {scene:<12s}  {'N/A':>8s}  {'N/A':>8s}  {'N/A':>8s}  {'N/A':>8s}")
            continue
        r = data.get("results", data)
        psnr = r.get("psnr", 0)
        ssim = r.get("ssim", 0)
        lpips = r.get("lpips", 0)
        fps = r.get("fps", 0)
        print(f"  {scene:<12s}  {psnr:>8.2f}  {ssim:>8.4f}  {lpips:>8.4f}  {fps:>8.1f}")
        for k in metrics_agg:
            if k in r:
                metrics_agg[k].append(r[k])

    print("-" * 70)
    n = len(metrics_agg["psnr"])
    if n > 0:
        avg = {k: sum(v) / len(v) for k, v in metrics_agg.items() if v}
        print(f"  {'AVERAGE':<12s}  {avg.get('psnr',0):>8.2f}  {avg.get('ssim',0):>8.4f}  "
              f"{avg.get('lpips',0):>8.4f}  {avg.get('fps',0):>8.1f}")
    print("=" * 70)
    return {k: sum(v) / len(v) for k, v in metrics_agg.items() if v}


def train_and_eval(scene, method, gpu, max_iters, eval_only, results_dir,
                   output_root, data_root, dataparser, dataparser_args):
    """Train and eval a single scene. Returns (scene, result_dict, tb_data)."""
    status = f"[GPU {gpu}] {scene}"
    tb_data = None
    if not eval_only:
        print(f"\n>>> {status}: TRAINING <<<")
        success, err = train_scene(
            scene, method, gpu, max_iters,
            output_root=output_root, data_root=data_root,
            dataparser=dataparser, dataparser_args=dataparser_args,
        )
        if not success:
            print(f"  {status}: Training FAILED, skipping eval.")
            return (scene, None, None)
        # Read PSNR from TensorBoard
        run_dir = get_latest_run_dir(scene, method, output_root)
        if run_dir:
            tb_data = read_psnr_from_tb(run_dir)

    print(f"\n>>> {status}: EVALUATING <<<")
    data = eval_scene(scene, method, gpu, results_dir, output_root=output_root)
    return (scene, data, tb_data)


def main():
    parser = argparse.ArgumentParser(description="Train & eval NeRF methods")
    parser.add_argument("--eval-only", action="store_true", help="Skip training, only evaluate")
    parser.add_argument("--gpu", type=str, default="0", help="GPU IDs, comma-separated")
    parser.add_argument("--method", type=str, required=True, help="Method name (CLI name)")
    parser.add_argument("--dataset", type=str, default="mipnerf360", choices=list(DATASETS.keys()))
    parser.add_argument("--scenes", nargs="+", default=None, help="Subset of scenes")
    parser.add_argument("--max-num-iterations", type=int, default=3000, help="Max training iterations")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    dataset_cfg = DATASETS[args.dataset]
    data_root = dataset_cfg["data_root"]
    dataparser = dataset_cfg["dataparser"]
    dataparser_args = dataset_cfg.get("dataparser_args", [])
    method = args.method
    gpus = [int(g.strip()) for g in args.gpu.split(",")]
    scenes = args.scenes if args.scenes else dataset_cfg["scenes"]
    output_root = Path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT_ROOT

    results_dir = output_root / "results" / method
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Dataset:    {args.dataset}")
    print(f"Data root:  {data_root}")
    print(f"Method:     {method}")
    print(f"GPUs:       {gpus}")
    print(f"Scenes:     {scenes}")
    print(f"Max iters:  {args.max_num_iterations}")
    print(f"Output:     {output_root}")

    all_results = {}
    all_tb = {}

    with ProcessPoolExecutor(max_workers=len(gpus)) as executor:
        futures = {}
        for i, scene in enumerate(scenes):
            gpu = gpus[i % len(gpus)]
            future = executor.submit(
                train_and_eval, scene, method, gpu, args.max_num_iterations,
                args.eval_only, results_dir, output_root, data_root,
                dataparser, dataparser_args,
            )
            futures[future] = scene

        for future in as_completed(futures):
            scene = futures[future]
            try:
                _, data, tb_data = future.result()
                all_results[scene] = data
                all_tb[scene] = tb_data
            except Exception as e:
                print(f"  ERROR: {scene} failed with: {e}")
                all_results[scene] = None

    averages = print_results_table(all_results, method)

    # Save summary
    summary = {
        "method": method,
        "dataset": args.dataset,
        "max_iters": args.max_num_iterations,
        "timestamp": datetime.now().isoformat(),
        "averages": averages,
        "per_scene": all_results,
        "tensorboard": {k: v for k, v in all_tb.items() if v is not None},
    }
    summary_path = results_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
