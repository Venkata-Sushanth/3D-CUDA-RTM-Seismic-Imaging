# 3D CUDA-Accelerated Reverse Time Migration (RTM) — Seismic Imaging

An educational parallel-computing project demonstrating 3D acoustic wave propagation and a basic Reverse Time Migration (RTM) imaging condition. The repository contains a sequential C++ CPU reference, a native CUDA C++ implementation, and a Google Colab notebook for compiling, running, and visualizing the GPU implementation.

## Project goal

Seismic imaging simulates wave propagation through underground rock layers and compares modeled fields with receiver recordings to help reconstruct subsurface structure. A 3D grid has many spatial points and must be updated across many time steps, making it a useful example of GPU parallel computing.

This prototype uses synthetic data to demonstrate the workflow. It is not a production seismic-processing tool and does not identify real reservoirs or faults.

## Features

- Synthetic 64 × 64 × 64 3D velocity model with layered regions, an offset layer boundary, and a small velocity anomaly.
- Ricker-wavelet source and 56 synthetic receiver locations.
- Forward 3D acoustic wave propagation over a series of time steps.
- Synthetic receiver trace recording.
- Reverse-time propagation and a basic forward/backward wavefield correlation image.
- CPU and CUDA implementations with timing output and CSV results.
- Colab notebook for compiling the CUDA source and plotting the outputs.

## Repository layout

    .
    ├── README.md
    ├── .gitignore
    ├── src/
    │   ├── cpu_rtm.cpp
    │   └── cuda_rtm.cu
    ├── notebooks/
    │   └── 3D_CUDA_RTM_Seismic_Imaging.ipynb
    └── results/
        └── .gitkeep

## Requirements

For the CPU reference, use a C++17 compiler such as g++.

For the CUDA version, use an NVIDIA GPU, a compatible NVIDIA driver, and the CUDA Toolkit with nvcc. The notebook is intended for a Colab GPU runtime where the CUDA compiler is available.

## Run the CPU reference

Run these commands from the repository root:

    g++ -O3 -std=c++17 src/cpu_rtm.cpp -o cpu_rtm
    ./cpu_rtm

The program prints forward-modeling and total runtime. It writes:

- results/cpu_velocity_model.csv
- results/cpu_receiver_gather.csv
- results/cpu_rtm_image.csv

## Run the CUDA implementation

Run on a system with a working CUDA Toolkit and NVIDIA GPU:

    nvcc -O3 -std=c++17 src/cuda_rtm.cu -o cuda_rtm
    ./cuda_rtm

The program reports the detected GPU and timing information. It writes:

- results/cuda_velocity_model.csv
- results/cuda_receiver_gather.csv
- results/cuda_rtm_image.csv

## Run with Google Colab

1. Open the notebook at notebooks/3D_CUDA_RTM_Seismic_Imaging.ipynb in Google Colab.
2. Select Runtime → Change runtime type → GPU, when available.
3. Run cells from top to bottom.
4. Review the printed runtime values and the generated model, receiver-gather, and RTM plots.

The notebook clones this public repository and compiles the native CUDA C++ source. The GPU type and availability depend on the runtime assigned to the session.

## Benchmarking responsibly

1. Use the same grid dimensions, time-step count, and model for CPU/GPU comparisons.
2. Run each version multiple times and report a representative timing; initial CUDA setup can affect the first run.
3. Compare numerical outputs as well as elapsed time. Small floating-point differences are possible.
4. Record the CPU, GPU, compiler, and CUDA Toolkit versions in the report.
5. Do not claim that a GPU is always faster. Performance depends on grid size, memory traffic, compiler settings, and hardware.

## Limitations

This implementation is a small demonstration using a second-order, 7-point finite-difference acoustic equation, synthetic receiver traces, fixed-zero outer boundaries, and a basic correlation imaging condition. More realistic RTM workflows require better absorbing boundaries, validated discretization and stability checks, realistic/prepared seismic data, improved imaging conditions, and benchmark validation. The output should not be interpreted as evidence of an actual geological reservoir or fault.

## License

No license has been added. Add a license file before explicitly permitting reuse under particular terms.
