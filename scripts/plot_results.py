#!/usr/bin/env python3
"""
scripts/plot_results.py
-----------------------
Generates publication-quality visualizations for the 3D CUDA Reverse Time Migration project.
Outputs saved to results/:
  - results/rtm_summary_dashboard.png (4-panel comprehensive figure)
  - results/velocity_model.png
  - results/receiver_gather.png
  - results/rtm_image_raw.png
  - results/rtm_image_laplacian.png
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import laplace


def compute_laplacian_3d(volume: np.ndarray, dx: float = 10.0) -> np.ndarray:
    """Computes a 3D finite-difference Laplacian to filter low-frequency RTM artifacts."""
    return laplace(volume) / (dx * dx)


def load_grid_csv(path: Path, nx: int = 64, ny: int = 64, nz: int = 64) -> np.ndarray:
    """Loads a 3D flattened grid CSV and reshapes it to (NZ, NY, NX)."""
    df = pd.read_csv(path)
    return df["value"].to_numpy(dtype=np.float32).reshape(nz, ny, nx)


def load_gather_csv(path: Path, nr: int = 56) -> np.ndarray:
    """Loads a receiver gather CSV and reshapes to (NT, NR)."""
    df = pd.read_csv(path)
    return df["amplitude"].to_numpy(dtype=np.float32).reshape(-1, nr)


def find_dataset(results_dir: Path, target: str):
    """Finds cuda or cpu variant for a given dataset."""
    cuda_file = results_dir / f"cuda_{target}.csv"
    cpu_file = results_dir / f"cpu_{target}.csv"
    if cuda_file.exists():
        return cuda_file, "CUDA"
    if cpu_file.exists():
        return cpu_file, "CPU"
    return None, None


def main():
    parser = argparse.ArgumentParser(description="Plot 3D RTM Seismic Imaging Results")
    parser.add_argument("--results-dir", type=str, default="results", help="Directory with CSV results")
    parser.add_argument("--prefix", type=str, choices=["cuda", "cpu", "auto"], default="auto",
                        help="Data prefix to plot (default: auto)")
    parser.add_argument("--slice-y", type=int, default=32, help="Y-index slice to display (default: 32)")
    parser.add_argument("--save-dpi", type=int, default=300, help="Image resolution DPI (default: 300)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    nx, ny, nz = 64, 64, 64
    dx, dt = 10.0, 0.001
    nr = 56
    y_idx = args.slice_y

    print("=" * 70)
    print(" 3D Seismic RTM Plotting and Visualization")
    print("=" * 70)

    # Resolve files
    prefix = args.prefix
    if prefix == "auto":
        v_file, source = find_dataset(results_dir, "velocity_model")
        g_file, _ = find_dataset(results_dir, "receiver_gather")
        raw_file, _ = find_dataset(results_dir, "rtm_image")
        lap_file, _ = find_dataset(results_dir, "rtm_image_laplacian")
    else:
        v_file = results_dir / f"{prefix}_velocity_model.csv"
        g_file = results_dir / f"{prefix}_receiver_gather.csv"
        raw_file = results_dir / f"{prefix}_rtm_image.csv"
        lap_file = results_dir / f"{prefix}_rtm_image_laplacian.csv"
        source = prefix.upper()

    if not v_file or not v_file.exists():
        print(f"[Error] Velocity model CSV not found in {results_dir.resolve()}.")
        print("Please run either ./cpu_rtm or ./cuda_rtm first to generate data.")
        sys.exit(1)

    print(f"Loading {source} results from: {results_dir.resolve()}")
    velocity = load_grid_csv(v_file, nx, ny, nz)
    traces = load_gather_csv(g_file, nr) if g_file and g_file.exists() else None
    rtm_raw = load_grid_csv(raw_file, nx, ny, nz) if raw_file and raw_file.exists() else None

    if lap_file and lap_file.exists():
        rtm_lap = load_grid_csv(lap_file, nx, ny, nz)
    elif rtm_raw is not None:
        print("Generating 3D Laplacian-filtered image on the fly...")
        rtm_lap = compute_laplacian_3d(rtm_raw, dx=dx)
    else:
        rtm_lap = None

    # Physical coordinate extents
    x_extent = [0, (nx - 1) * dx, (nz - 1) * dx, 0]  # [x_min, x_max, z_max, z_min] for origin='upper'
    nt = traces.shape[0] if traces is not None else 240
    t_extent = [4 * dx, (4 + nr - 1) * dx, (nt - 1) * dt * 1000, 0]  # [x_min, x_max, t_max_ms, t_min_ms]

    # Style configuration
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig = plt.figure(figsize=(18, 12), dpi=args.save_dpi)
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.25)

    # 1. Velocity Model
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(velocity[:, y_idx, :], extent=x_extent, cmap="viridis", aspect="auto", origin="upper")
    ax1.plot(32 * dx, 3 * dx, "r*", markersize=14, markeredgecolor="black", label="Source Shot (Sx, Sz)")
    ax1.scatter(np.arange(4, 4 + nr) * dx, np.full(nr, 3 * dx), s=15, c="cyan", edgecolors="black",
                marker="v", label=f"Receiver Array ({nr} geophones)", zorder=3)
    ax1.set_title(f"(A) Subsurface Velocity Model (Cross-section Y = {y_idx * dx:.0f} m)", fontsize=13, weight="bold")
    ax1.set_xlabel("Horizontal Distance X (meters)", fontsize=11)
    ax1.set_ylabel("Depth Z (meters)", fontsize=11)
    ax1.legend(loc="lower left", framealpha=0.9)
    cb1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cb1.set_label("Acoustic Velocity (m/s)", fontsize=10)

    # 2. Receiver Gather (Shot Seismogram)
    ax2 = fig.add_subplot(gs[0, 1])
    if traces is not None:
        limit_t = np.percentile(np.abs(traces), 99.0) + 1e-12
        im2 = ax2.imshow(traces, extent=t_extent, cmap="seismic", aspect="auto",
                         vmin=-limit_t, vmax=limit_t, origin="upper")
        ax2.set_title(f"(B) Synthetic Common-Shot Gather ({source})", fontsize=13, weight="bold")
        ax2.set_xlabel("Receiver Coordinate X (meters)", fontsize=11)
        ax2.set_ylabel("Two-Way Travel Time (ms)", fontsize=11)
        cb2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        cb2.set_label("Pressure Amplitude", fontsize=10)
    else:
        ax2.text(0.5, 0.5, "Receiver Gather Data Not Available", ha="center", va="center")

    # 3. Raw RTM Image
    ax3 = fig.add_subplot(gs[1, 0])
    if rtm_raw is not None:
        slice_raw = rtm_raw[:, y_idx, :]
        limit_r = np.percentile(np.abs(slice_raw), 99.5) + 1e-12
        im3 = ax3.imshow(slice_raw, extent=x_extent, cmap="seismic", aspect="auto",
                         vmin=-limit_r, vmax=limit_r, origin="upper")
        ax3.set_title(f"(C) Standard RTM Image (Cross-correlation, Y = {y_idx * dx:.0f} m)", fontsize=13, weight="bold")
        ax3.set_xlabel("Horizontal Distance X (meters)", fontsize=11)
        ax3.set_ylabel("Depth Z (meters)", fontsize=11)
        cb3 = fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
        cb3.set_label("Correlation Energy", fontsize=10)
    else:
        ax3.text(0.5, 0.5, "Raw RTM Image Not Available", ha="center", va="center")

    # 4. Laplacian-Filtered RTM Image (Clean Reflectors)
    ax4 = fig.add_subplot(gs[1, 1])
    if rtm_lap is not None:
        slice_lap = rtm_lap[:, y_idx, :]
        limit_l = np.percentile(np.abs(slice_lap), 99.5) + 1e-12
        im4 = ax4.imshow(slice_lap, extent=x_extent, cmap="seismic", aspect="auto",
                         vmin=-limit_l, vmax=limit_l, origin="upper")
        ax4.set_title(f"(D) Filtered RTM Image (Laplacian Filtered - Sharp Interfaces)", fontsize=13, weight="bold")
        ax4.set_xlabel("Horizontal Distance X (meters)", fontsize=11)
        ax4.set_ylabel("Depth Z (meters)", fontsize=11)
        cb4 = fig.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
        cb4.set_label("High-Pass Reflectivity", fontsize=10)
    else:
        ax4.text(0.5, 0.5, "Laplacian RTM Image Not Available", ha="center", va="center")

    fig.suptitle(f"3D CUDA Reverse Time Migration (RTM) Analysis Dashboard [{source}]",
                 fontsize=16, weight="bold", y=0.98)

    dashboard_path = results_dir / "rtm_summary_dashboard.png"
    plt.savefig(dashboard_path, bbox_inches="tight", dpi=args.save_dpi)
    plt.close()
    print(f"[Saved] Unified dashboard: {dashboard_path}")

    # Save individual high-resolution figures for reports/presentations
    def save_single(data, title, xlabel, ylabel, cmap, extent, out_name, vmin=None, vmax=None, clabel=""):
        fig_s, ax_s = plt.subplots(figsize=(8, 6), dpi=args.save_dpi)
        im_s = ax_s.imshow(data, extent=extent, cmap=cmap, aspect="auto", origin="upper", vmin=vmin, vmax=vmax)
        ax_s.set_title(title, fontsize=13, weight="bold")
        ax_s.set_xlabel(xlabel, fontsize=11)
        ax_s.set_ylabel(ylabel, fontsize=11)
        cb = fig_s.colorbar(im_s, ax=ax_s, fraction=0.046, pad=0.04)
        if clabel:
            cb.set_label(clabel, fontsize=10)
        p = results_dir / out_name
        fig_s.savefig(p, bbox_inches="tight", dpi=args.save_dpi)
        plt.close(fig_s)
        print(f"[Saved] Single plot: {p}")

    save_single(velocity[:, y_idx, :], f"Subsurface Velocity Model (Y = {y_idx * dx:.0f} m)",
                "Distance X (m)", "Depth Z (m)", "viridis", x_extent, "velocity_model.png", clabel="Velocity (m/s)")

    if traces is not None:
        save_single(traces, f"Synthetic Receiver Gather ({source})", "Receiver X (m)",
                    "Two-Way Travel Time (ms)", "seismic", t_extent, "receiver_gather.png",
                    vmin=-limit_t, vmax=limit_t, clabel="Amplitude")

    if rtm_raw is not None:
        save_single(rtm_raw[:, y_idx, :], f"Raw RTM Correlation Image ({source})", "Distance X (m)",
                    "Depth Z (m)", "seismic", x_extent, "rtm_image_raw.png",
                    vmin=-limit_r, vmax=limit_r, clabel="Correlation")

    if rtm_lap is not None:
        save_single(rtm_lap[:, y_idx, :], f"Laplacian-Filtered RTM Image ({source})", "Distance X (m)",
                    "Depth Z (m)", "seismic", x_extent, "rtm_image_laplacian.png",
                    vmin=-limit_l, vmax=limit_l, clabel="Reflectivity")

    print("[SUCCESS] All plots generated and saved successfully.")


if __name__ == "__main__":
    main()
