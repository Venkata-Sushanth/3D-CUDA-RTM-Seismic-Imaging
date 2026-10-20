#!/usr/bin/env python3
"""
scripts/benchmark.py
--------------------
Automates compilation, execution, and performance benchmarking between
the sequential CPU reference and the CUDA GPU implementation.
Measures:
  - Forward modeling elapsed time (ms)
  - Total RTM migration elapsed time (ms)
  - Speedup factor (CPU / GPU)
  - Automatic invocation of verify_results.py and plot_results.py
"""

import sys
import os
import re
import time
import subprocess
import argparse
from pathlib import Path


def parse_output(output: str):
    """Parses forward and total times from program output."""
    fwd_match = re.search(r"Forward modeling:\s*([\d\.]+)\s*ms", output)
    total_match = re.search(r"Forward \+ RTM imaging:\s*([\d\.]+)\s*ms", output)
    fwd_ms = float(fwd_match.group(1)) if fwd_match else None
    total_ms = float(total_match.group(1)) if total_match else None
    return fwd_ms, total_ms


def run_command(cmd, desc):
    """Executes a command and returns (success, stdout, stderr, elapsed_wall_s)."""
    print(f"--> [{desc}]: {' '.join(str(c) for c in cmd)}")
    t0 = time.perf_counter()
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    dt = time.perf_counter() - t0
    if res.returncode != 0:
        print(f"Error ({desc}):\n{res.stderr}")
        return False, res.stdout, res.stderr, dt
    return True, res.stdout, res.stderr, dt


def main():
    parser = argparse.ArgumentParser(description="Benchmark 3D RTM: CPU vs CUDA")
    parser.add_argument("--skip-compile", action="store_true", help="Skip compilation step")
    parser.add_argument("--skip-verify", action="store_true", help="Skip numerical verification")
    parser.add_argument("--skip-plot", action="store_true", help="Skip plotting step")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    os.chdir(project_root)

    cpu_bin = project_root / ("cpu_rtm.exe" if os.name == "nt" else "cpu_rtm")
    cuda_bin = project_root / ("cuda_rtm.exe" if os.name == "nt" else "cuda_rtm")

    print("=" * 75)
    print(" 3D Reverse Time Migration (RTM) Benchmark Suite")
    print("=" * 75)

    # 1. Compilation
    if not args.skip_compile:
        # Check g++
        try:
            subprocess.run(["g++", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            run_command(["g++", "-O3", "-std=c++17", "src/cpu_rtm.cpp", "-o", str(cpu_bin)], "Compiling CPU reference")
        except (subprocess.SubprocessError, FileNotFoundError):
            print("[Info] g++ compiler not found. Skipping CPU compilation.")

        # Check nvcc
        try:
            subprocess.run(["nvcc", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            run_command(["nvcc", "-O3", "-std=c++17", "src/cuda_rtm.cu", "-o", str(cuda_bin)], "Compiling CUDA source")
        except (subprocess.SubprocessError, FileNotFoundError):
            print("[Info] nvcc compiler not found. Skipping CUDA compilation.")

    # 2. Run CPU
    cpu_fwd, cpu_total = None, None
    if cpu_bin.exists():
        ok, out, _, _ = run_command([str(cpu_bin)], "Running CPU Reference")
        if ok:
            cpu_fwd, cpu_total = parse_output(out)
            print(f"    CPU Forward: {cpu_fwd:.1f} ms | Total: {cpu_total:.1f} ms")
    else:
        print(f"[-] CPU executable not found at {cpu_bin}")

    # 3. Run CUDA
    cuda_fwd, cuda_total = None, None
    if cuda_bin.exists():
        ok, out, _, _ = run_command([str(cuda_bin)], "Running CUDA Implementation")
        if ok:
            cuda_fwd, cuda_total = parse_output(out)
            print(f"    CUDA Forward: {cuda_fwd:.1f} ms | Total: {cuda_total:.1f} ms")
    else:
        print(f"[-] CUDA executable not found at {cuda_bin}")

    # 4. Summary Table & Speedup
    print("\n" + "=" * 75)
    print(" PERFORMANCE BENCHMARK SUMMARY")
    print("=" * 75)
    print(f"{'Implementation':<22} | {'Forward Time (ms)':<18} | {'Total RTM Time (ms)':<18}")
    print("-" * 75)
    if cpu_total:
        print(f"{'CPU Sequential':<22} | {cpu_fwd:<18.2f} | {cpu_total:<18.2f}")
    else:
        print(f"{'CPU Sequential':<22} | {'N/A':<18} | {'N/A':<18}")

    if cuda_total:
        print(f"{'CUDA Accelerated':<22} | {cuda_fwd:<18.2f} | {cuda_total:<18.2f}")
    else:
        print(f"{'CUDA Accelerated':<22} | {'N/A':<18} | {'N/A':<18}")

    print("-" * 75)
    if cpu_total and cuda_total and cuda_total > 0:
        speedup_fwd = cpu_fwd / cuda_fwd if (cpu_fwd and cuda_fwd) else 0.0
        speedup_tot = cpu_total / cuda_total
        print(f"[*] FORWARD SPEEDUP: {speedup_fwd:.2f}x")
        print(f"[*] TOTAL RTM SPEEDUP: {speedup_tot:.2f}x")
    print("=" * 75)

    # 5. Parity Check
    if not args.skip_verify:
        print("\n--> Running numerical parity verification...")
        subprocess.run([sys.executable, "scripts/verify_results.py"], check=False)

    # 6. Plotting
    if not args.skip_plot:
        print("\n--> Generating result plots...")
        subprocess.run([sys.executable, "scripts/plot_results.py"], check=False)


if __name__ == "__main__":
    main()
