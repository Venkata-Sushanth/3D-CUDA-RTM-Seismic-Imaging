#!/usr/bin/env python3
"""
scripts/verify_results.py
-------------------------
Verifies numerical parity between the sequential CPU reference and CUDA GPU implementations.
Compares:
  - Velocity models (cpu_velocity_model.csv vs cuda_velocity_model.csv)
  - Receiver gathers (cpu_receiver_gather.csv vs cuda_receiver_gather.csv)
  - Raw RTM images (cpu_rtm_image.csv vs cuda_rtm_image.csv)
  - Laplacian-filtered RTM images (cpu_rtm_image_laplacian.csv vs cuda_rtm_image_laplacian.csv)
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def compute_metrics(a: np.ndarray, b: np.ndarray):
    """Computes comparison metrics between two arrays."""
    diff = np.abs(a - b)
    max_abs = float(np.max(diff))
    mae = float(np.mean(diff))
    
    norm_a = float(np.linalg.norm(a))
    if norm_a > 1e-12:
        rel_l2 = float(np.linalg.norm(diff) / norm_a)
    else:
        rel_l2 = 0.0

    # Pearson correlation coefficient
    std_a, std_b = np.std(a), np.std(b)
    if std_a > 1e-12 and std_b > 1e-12:
        corr = float(np.corrcoef(a.flatten(), b.flatten())[0, 1])
    else:
        corr = 1.0 if max_abs < 1e-6 else 0.0

    return max_abs, mae, rel_l2, corr


def verify_pair(cpu_path: Path, cuda_path: Path, val_col: str, tol_rel: float = 1e-3):
    """Loads and verifies a single CPU-CUDA output pair."""
    if not cpu_path.exists():
        return False, f"Missing CPU file: {cpu_path.name}"
    if not cuda_path.exists():
        return False, f"Missing CUDA file: {cuda_path.name}"

    try:
        df_cpu = pd.read_csv(cpu_path)
        df_cuda = pd.read_csv(cuda_path)
    except Exception as e:
        return False, f"Error reading CSV files: {e}"

    if val_col not in df_cpu.columns or val_col not in df_cuda.columns:
        return False, f"Column '{val_col}' missing from datasets"

    if len(df_cpu) != len(df_cuda):
        return False, f"Row count mismatch: CPU={len(df_cpu)}, CUDA={len(df_cuda)}"

    arr_cpu = df_cpu[val_col].to_numpy(dtype=np.float32)
    arr_cuda = df_cuda[val_col].to_numpy(dtype=np.float32)

    max_abs, mae, rel_l2, corr = compute_metrics(arr_cpu, arr_cuda)
    passed = rel_l2 <= tol_rel and corr >= 0.999

    return passed, {
        "max_abs": max_abs,
        "mae": mae,
        "rel_l2": rel_l2,
        "corr": corr,
        "size": len(arr_cpu)
    }


def main():
    parser = argparse.ArgumentParser(description="Verify CPU vs CUDA RTM Numerical Parity")
    parser.add_argument("--results-dir", type=str, default="results", help="Directory containing CSV outputs")
    parser.add_argument("--tolerance", type=float, default=1e-3, help="Relative L2 tolerance (default: 1e-3)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    comparisons = [
        ("Velocity Model", "cpu_velocity_model.csv", "cuda_velocity_model.csv", "value", 1e-5),
        ("Receiver Gather", "cpu_receiver_gather.csv", "cuda_receiver_gather.csv", "amplitude", args.tolerance),
        ("Raw RTM Image", "cpu_rtm_image.csv", "cuda_rtm_image.csv", "value", args.tolerance),
        ("Laplacian RTM Image", "cpu_rtm_image_laplacian.csv", "cuda_rtm_image_laplacian.csv", "value", args.tolerance),
    ]

    print("=" * 80)
    print(" 3D RTM Numerical Parity Verification (CPU vs CUDA)")
    print("=" * 80)
    print(f"Results Directory: {results_dir.resolve()}")
    print("-" * 80)

    found_any = False
    all_passed = True

    for name, cpu_file, cuda_file, col, tol in comparisons:
        cpu_path = results_dir / cpu_file
        cuda_path = results_dir / cuda_file

        if not cpu_path.exists() and not cuda_path.exists():
            print(f"[-] {name:<22} : Neither CPU nor CUDA file found.")
            continue

        found_any = True
        passed, res = verify_pair(cpu_path, cuda_path, col, tol_rel=tol)

        if not passed and isinstance(res, str):
            print(f"[!] {name:<22} : {res}")
            all_passed = False
            continue

        status_str = "[PASS]" if passed else "[FAIL]"
        if not passed:
            all_passed = False

        print(f"{status_str} {name:<20} | Rel L2: {res['rel_l2']:.2e} | Max Diff: {res['max_abs']:.2e} | "
              f"Corr: {res['corr']:.6f} | Count: {res['size']}")

    print("=" * 80)
    if not found_any:
        print("Note: No comparison files were found in the results directory.")
        print("Run the simulation executables first:")
        print("  1. ./cpu_rtm   (or g++ -O3 -std=c++17 src/cpu_rtm.cpp -o cpu_rtm && ./cpu_rtm)")
        print("  2. ./cuda_rtm  (or nvcc -O3 -std=c++17 src/cuda_rtm.cu -o cuda_rtm && ./cuda_rtm)")
        sys.exit(0)

    if all_passed:
        print("[SUCCESS] All available output pairs passed parity checks within numerical tolerance.")
        sys.exit(0)
    else:
        print("[WARNING] One or more comparisons failed parity verification or are missing files.")
        sys.exit(1)


if __name__ == "__main__":
    main()
