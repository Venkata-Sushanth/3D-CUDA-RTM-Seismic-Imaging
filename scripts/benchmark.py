#!/usr/bin/env python3
"""
Benchmark sequential CPU and parallel CUDA implementations of 3D RTM.

The benchmark reports the execution times printed by each executable:
  - Forward modeling time
  - Total RTM time
It repeats each implementation, reports mean and standard deviation, and
calculates GPU speedup and execution-time reduction relative to the CPU.

Important: CPU and CUDA timing boundaries must cover identical operations
for the total-time comparison to be scientifically fair. See the warning
printed in the report and align src/cpu_rtm.cpp / src/cuda_rtm.cu if needed.
"""

import argparse
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path


def parse_output(output: str):
    """Return (forward_ms, total_ms), or raise if timings are missing."""
    forward = re.search(r"Forward modeling:\s*([\d.]+)\s*ms", output)
    total = re.search(r"Forward \+ RTM imaging:\s*([\d.]+)\s*ms", output)

    if not forward or not total:
        raise ValueError(
            "Could not find both timing lines in program output. "
            "Expected 'Forward modeling: ... ms' and "
            "'Forward + RTM imaging: ... ms'."
        )

    return float(forward.group(1)), float(total.group(1))


def run_command(cmd, description, quiet=False):
    """Run a command and return (success, stdout, stderr, wall_seconds)."""
    if not quiet:
        print(f"[RUN] {description}: {' '.join(map(str, cmd))}")

    start = time.perf_counter()
    result = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    elapsed = time.perf_counter() - start

    if result.returncode != 0:
        print(f"\n[ERROR] {description} failed (exit code {result.returncode}).")
        if result.stderr.strip():
            print(result.stderr.strip())
        if result.stdout.strip():
            print(result.stdout.strip())
        return False, result.stdout, result.stderr, elapsed

    return True, result.stdout, result.stderr, elapsed


def compile_programs(cpu_bin, cuda_bin):
    """Compile CPU and CUDA programs. Stop if a required compiler/build fails."""
    builds = [
        ("CPU sequential implementation", ["g++", "-O3", "-std=c++17",
                                            "src/cpu_rtm.cpp", "-o", str(cpu_bin)]),
        ("CUDA parallel implementation", ["nvcc", "-O3", "-std=c++17",
                                           "src/cuda_rtm.cu", "-o", str(cuda_bin)]),
    ]

    for label, command in builds:
        compiler = command[0]
        try:
            subprocess.run(
                [compiler, "--version"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )
        except (FileNotFoundError, subprocess.CalledProcessError):
            print(f"[ERROR] Required compiler '{compiler}' was not found.")
            return False

        ok, _, _, _ = run_command(command, f"Compiling {label}")
        if not ok:
            return False

    return True


def collect_runs(binary, label, runs):
    """Execute one implementation repeatedly and collect program-reported ms."""
    forward_times = []
    total_times = []

    print(f"\n[MEASURING] {label} — {runs} run(s)")
    for run_number in range(1, runs + 1):
        ok, output, _, wall_seconds = run_command(
            [str(binary)], f"{label}, run {run_number}/{runs}", quiet=True
        )
        if not ok:
            return None

        try:
            forward_ms, total_ms = parse_output(output)
        except ValueError as error:
            print(f"[ERROR] {label}, run {run_number}: {error}")
            print("----- Program output -----")
            print(output)
            print("--------------------------")
            return None

        forward_times.append(forward_ms)
        total_times.append(total_ms)
        print(
            f"  Run {run_number:>2}: "
            f"Forward = {forward_ms:>10.3f} ms | "
            f"Total RTM = {total_ms:>10.3f} ms "
            f"| Process wall time = {wall_seconds:>8.3f} s"
        )

    return {
        "forward": forward_times,
        "total": total_times,
    }


def mean_and_std(values):
    """Return mean and sample standard deviation (0 for a single run)."""
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, std


def print_metric_row(label, cpu_values, gpu_values):
    cpu_mean, cpu_std = mean_and_std(cpu_values)
    gpu_mean, gpu_std = mean_and_std(gpu_values)

    saved_ms = cpu_mean - gpu_mean
    speedup = cpu_mean / gpu_mean if gpu_mean > 0 else float("inf")
    reduction = (saved_ms / cpu_mean * 100.0) if cpu_mean > 0 else 0.0

    print(f"\n{label}")
    print("-" * 79)
    print(f"  CPU Sequential : {cpu_mean:10.3f} ms  (± {cpu_std:.3f} ms)")
    print(f"  GPU Parallel   : {gpu_mean:10.3f} ms  (± {gpu_std:.3f} ms)")
    print(f"  Time Saved     : {saved_ms:10.3f} ms")
    print(f"  Speedup        : {speedup:10.2f}x")
    print(f"  Time Reduction : {reduction:9.2f}%")
    return cpu_mean, gpu_mean, speedup, reduction


def main():
    parser = argparse.ArgumentParser(
        description="Compare sequential CPU and parallel CUDA 3D RTM execution times."
    )
    parser.add_argument(
        "--runs", type=int, default=5,
        help="Number of benchmark runs per implementation (default: 5)."
    )
    parser.add_argument("--skip-compile", action="store_true",
                        help="Use existing CPU and CUDA executables.")
    parser.add_argument("--skip-verify", action="store_true",
                        help="Skip numerical result verification.")
    parser.add_argument("--skip-plot", action="store_true",
                        help="Skip result plotting.")
    args = parser.parse_args()

    if args.runs < 1:
        parser.error("--runs must be at least 1")

    project_root = Path(__file__).resolve().parent.parent
    os.chdir(project_root)

    cpu_bin = project_root / ("cpu_rtm.exe" if os.name == "nt" else "cpu_rtm")
    cuda_bin = project_root / ("cuda_rtm.exe" if os.name == "nt" else "cuda_rtm")

    print("=" * 79)
    print("       3D CUDA-ACCELERATED REVERSE TIME MIGRATION (RTM)")
    print("              SEQUENTIAL CPU vs PARALLEL GPU")
    print("=" * 79)
    print("\n[1] BENCHMARK CONFIGURATION")
    print("-" * 79)
    print(f"  CPU implementation : Sequential C++")
    print(f"  GPU implementation : Parallel CUDA")
    print(f"  Number of runs     : {args.runs}")
    print(f"  Timing source      : Timings reported by the CPU/CUDA programs")
    print("-" * 79)

    if not args.skip_compile:
        if not compile_programs(cpu_bin, cuda_bin):
            sys.exit(1)

    if not cpu_bin.exists():
        print(f"[ERROR] CPU executable not found: {cpu_bin}")
        sys.exit(1)
    if not cuda_bin.exists():
        print(f"[ERROR] CUDA executable not found: {cuda_bin}")
        sys.exit(1)

    cpu_results = collect_runs(cpu_bin, "CPU SEQUENTIAL", args.runs)
    if cpu_results is None:
        sys.exit(1)

    gpu_results = collect_runs(cuda_bin, "GPU PARALLEL (CUDA)", args.runs)
    if gpu_results is None:
        sys.exit(1)

    print("\n" + "=" * 79)
    print("                 CPU vs GPU EXECUTION TIME REPORT")
    print("=" * 79)
    print("All values below are averages across the requested benchmark runs.")
    print_metric_row(
        "[2] FORWARD MODELING EXECUTION TIME",
        cpu_results["forward"], gpu_results["forward"]
    )
    cpu_total, gpu_total, total_speedup, total_reduction = print_metric_row(
        "[3] TOTAL RTM EXECUTION TIME",
        cpu_results["total"], gpu_results["total"]
    )

    print("\n" + "=" * 79)
    print("                         FINAL RESULT")
    print("=" * 79)
    print(f"  Sequential CPU total time : {cpu_total:.3f} ms")
    print(f"  Parallel GPU total time   : {gpu_total:.3f} ms")
    print(f"  Total execution time saved: {cpu_total - gpu_total:.3f} ms")
    print(f"  Overall speedup           : {total_speedup:.2f}x")
    print(f"  Execution time reduction  : {total_reduction:.2f}%")
    print("=" * 79)

    print("\n[IMPORTANT COMPARISON NOTE]")
    print(
        "Ensure src/cpu_rtm.cpp and src/cuda_rtm.cu measure the same operations "
        "inside their total RTM timers. In the current source, the CUDA total "
        "includes Laplacian filtering while the CPU total timer ends before "
        "that operation. Until those timing boundaries are aligned, treat the "
        "total-time speedup as provisional."
    )

    if not args.skip_verify:
        print("\n[4] NUMERICAL CORRECTNESS VERIFICATION")
        print("-" * 79)
        verify = subprocess.run(
            [sys.executable, "scripts/verify_results.py"], check=False
        )
        if verify.returncode != 0:
            print("[WARNING] Numerical verification reported a failure.")

    if not args.skip_plot:
        print("\n[5] GENERATING RESULT PLOTS")
        print("-" * 79)
        plot = subprocess.run(
            [sys.executable, "scripts/plot_results.py"], check=False
        )
        if plot.returncode != 0:
            print("[WARNING] Plot generation reported a failure.")

    print("\n[COMPLETE] Benchmark process finished.")


if __name__ == "__main__":
    main()
