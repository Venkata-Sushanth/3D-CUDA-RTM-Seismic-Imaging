#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr int NX = 64, NY = 64, NZ = 64;
constexpr int NT = 240, NR = 56;
constexpr float DX = 10.0f, DT = 0.001f, F0 = 20.0f;
constexpr float PI = 3.14159265358979323846f;
constexpr std::size_t CELLS = static_cast<std::size_t>(NX) * NY * NZ;
constexpr std::size_t SNAPSHOT_VALUES = static_cast<std::size_t>(NT) * CELLS;
constexpr int SX = NX / 2, SY = NY / 2, SZ = 3;
constexpr int SPONGE = 8;
constexpr float ALPHA = 0.015f;

#define CUDA_CHECK(call) do { \
    cudaError_t err__ = (call); \
    if (err__ != cudaSuccess) throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err__)); \
} while (0)

inline std::size_t index3(int x, int y, int z) {
    return (static_cast<std::size_t>(z) * NY + y) * NX + x;
}
float ricker(int t) {
    const float a = PI * F0 * (t * DT - 1.5f / F0);
    const float a2 = a * a;
    return (1.0f - 2.0f * a2) * std::exp(-a2);
}
void build_damping(std::vector<float>& damping) {
    for (int z = 0; z < NZ; ++z) for (int y = 0; y < NY; ++y) for (int x = 0; x < NX; ++x) {
        const int dist_x = std::min(x, NX - 1 - x);
        const int dist_y = std::min(y, NY - 1 - y);
        const int dist_z = std::min(z, NZ - 1 - z);
        const int min_dist = std::min({dist_x, dist_y, dist_z});
        float factor = 1.0f;
        if (min_dist < SPONGE) {
            const float d = static_cast<float>(SPONGE - min_dist);
            factor = std::exp(-ALPHA * d * d);
        }
        damping[index3(x, y, z)] = factor;
    }
}
void build_velocity(std::vector<float>& velocity) {
    for (int z = 0; z < NZ; ++z) for (int y = 0; y < NY; ++y) for (int x = 0; x < NX; ++x) {
        const int offset = (x >= NX / 2) ? 4 : 0;
        float v = z < 20 + offset ? 1800.0f : z < 40 + offset ? 2300.0f : 2800.0f;
        const int dx = x - 43, dy = y - 35, dz = z - 48;
        if (dx * dx + dy * dy + dz * dz < 7 * 7) v = 3200.0f;
        velocity[index3(x, y, z)] = v;
    }
}

__global__ void wave_step(const float* previous, const float* current, float* next,
                          const float* velocity, const float* damping,
                          float source_amplitude, bool inject_source) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= NX || y >= NY || z >= NZ) return;
    const std::size_t i = (static_cast<std::size_t>(z) * NY + y) * NX + x;
    if (x == 0 || y == 0 || z == 0 || x == NX - 1 || y == NY - 1 || z == NZ - 1) {
        next[i] = 0.0f;
        return;
    }
    const float c = current[i];
    const float lap = (current[i - 1] + current[i + 1]
                     + current[i - NX] + current[i + NX]
                     + current[i - static_cast<std::size_t>(NX) * NY]
                     + current[i + static_cast<std::size_t>(NX) * NY]
                     - 6.0f * c) / (DX * DX);
    next[i] = (2.0f * c - previous[i] + velocity[i] * velocity[i] * DT * DT * lap) * damping[i];
    if (inject_source && x == SX && y == SY && z == SZ) next[i] += source_amplitude;
}

__global__ void save_snapshot(const float* field, float* snapshots, int t) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < CELLS) snapshots[static_cast<std::size_t>(t) * CELLS + i] = field[i];
}
__global__ void sample_receivers(const float* field, float* traces, int t) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < NR) {
        const int x = 4 + r;
        const std::size_t i = (static_cast<std::size_t>(SZ) * NY + NY / 2) * NX + x;
        traces[static_cast<std::size_t>(t) * NR + r] = field[i];
    }
}
__global__ void inject_receivers(float* field, const float* traces, int t) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < NR) {
        const int x = 4 + r;
        const std::size_t i = (static_cast<std::size_t>(SZ) * NY + NY / 2) * NX + x;
        field[i] += traces[static_cast<std::size_t>(t) * NR + r];
    }
}
__global__ void imaging_condition(const float* forward_snapshot, const float* backward,
                                 float* image) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < CELLS) image[i] += forward_snapshot[i] * backward[i];
}
__global__ void apply_laplacian_kernel(const float* input, float* output) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= NX || y >= NY || z >= NZ) return;
    const std::size_t i = (static_cast<std::size_t>(z) * NY + y) * NX + x;
    if (x == 0 || y == 0 || z == 0 || x == NX - 1 || y == NY - 1 || z == NZ - 1) {
        output[i] = 0.0f;
        return;
    }
    const float inv_dx2 = 1.0f / (DX * DX);
    output[i] = (input[i - 1] + input[i + 1]
               + input[i - NX] + input[i + NX]
               + input[i - static_cast<std::size_t>(NX) * NY]
               + input[i + static_cast<std::size_t>(NX) * NY]
               - 6.0f * input[i]) * inv_dx2;
}

void write_grid_csv(const std::string& path, const std::vector<float>& grid) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("Cannot open output file: " + path);
    out << "x,y,z,value\n" << std::setprecision(7);
    for (int z = 0; z < NZ; ++z) for (int y = 0; y < NY; ++y) for (int x = 0; x < NX; ++x)
        out << x << ',' << y << ',' << z << ',' << grid[index3(x, y, z)] << '\n';
}
void write_traces_csv(const std::string& path, const std::vector<float>& traces) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("Cannot open output file: " + path);
    out << "time_step,time_seconds,receiver,amplitude\n" << std::setprecision(7);
    for (int t = 0; t < NT; ++t) for (int r = 0; r < NR; ++r)
        out << t << ',' << t * DT << ',' << r << ',' << traces[static_cast<std::size_t>(t) * NR + r] << '\n';
}
}

int main() {
    try {
        int device_count = 0;
        CUDA_CHECK(cudaGetDeviceCount(&device_count));
        if (device_count == 0) throw std::runtime_error("No CUDA-capable GPU detected.");
        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
        CUDA_CHECK(cudaSetDevice(0));
        std::filesystem::create_directories("results");

        std::vector<float> velocity(CELLS), damping(CELLS, 1.0f);
        std::vector<float> host_traces(static_cast<std::size_t>(NT) * NR);
        std::vector<float> host_image(CELLS), host_image_lap(CELLS);
        build_velocity(velocity);
        build_damping(damping);

        float *d_velocity = nullptr, *d_damping = nullptr;
        float *d_prev = nullptr, *d_curr = nullptr, *d_next = nullptr;
        float *d_snapshots = nullptr, *d_traces = nullptr, *d_image = nullptr, *d_image_lap = nullptr;
        const std::size_t volume_bytes = CELLS * sizeof(float);
        const std::size_t snapshots_bytes = SNAPSHOT_VALUES * sizeof(float);
        const std::size_t traces_bytes = static_cast<std::size_t>(NT) * NR * sizeof(float);
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_velocity), volume_bytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_damping), volume_bytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_prev), volume_bytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_curr), volume_bytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_next), volume_bytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_snapshots), snapshots_bytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_traces), traces_bytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_image), volume_bytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_image_lap), volume_bytes));

        CUDA_CHECK(cudaMemcpy(d_velocity, velocity.data(), volume_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_damping, damping.data(), volume_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_prev, 0, volume_bytes));
        CUDA_CHECK(cudaMemset(d_curr, 0, volume_bytes));
        CUDA_CHECK(cudaMemset(d_next, 0, volume_bytes));
        CUDA_CHECK(cudaMemset(d_image, 0, volume_bytes));

        const dim3 block(8, 8, 4);
        const dim3 grid((NX + block.x - 1) / block.x,
                        (NY + block.y - 1) / block.y,
                        (NZ + block.z - 1) / block.z);
        const int linear_blocks = static_cast<int>((CELLS + 255) / 256);
        const auto total_start = std::chrono::steady_clock::now();
        const auto forward_start = std::chrono::steady_clock::now();
        for (int t = 0; t < NT; ++t) {
            wave_step<<<grid, block>>>(d_prev, d_curr, d_next, d_velocity, d_damping, ricker(t), true);
            save_snapshot<<<linear_blocks, 256>>>(d_next, d_snapshots, t);
            sample_receivers<<<1, 128>>>(d_next, d_traces, t);
            CUDA_CHECK(cudaGetLastError());
            std::swap(d_prev, d_curr);
            std::swap(d_curr, d_next);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        const auto forward_end = std::chrono::steady_clock::now();

        CUDA_CHECK(cudaMemset(d_prev, 0, volume_bytes));
        CUDA_CHECK(cudaMemset(d_curr, 0, volume_bytes));
        CUDA_CHECK(cudaMemset(d_next, 0, volume_bytes));
        for (int t = NT - 1; t >= 0; --t) {
            wave_step<<<grid, block>>>(d_prev, d_curr, d_next, d_velocity, d_damping, 0.0f, false);
            inject_receivers<<<1, 128>>>(d_next, d_traces, t);
            imaging_condition<<<linear_blocks, 256>>>(d_snapshots + static_cast<std::size_t>(t) * CELLS,
                                                      d_next, d_image);
            CUDA_CHECK(cudaGetLastError());
            std::swap(d_prev, d_curr);
            std::swap(d_curr, d_next);
        }
        apply_laplacian_kernel<<<grid, block>>>(d_image, d_image_lap);
        CUDA_CHECK(cudaDeviceSynchronize());
        const auto total_end = std::chrono::steady_clock::now();

        CUDA_CHECK(cudaMemcpy(host_traces.data(), d_traces, traces_bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(host_image.data(), d_image, volume_bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(host_image_lap.data(), d_image_lap, volume_bytes, cudaMemcpyDeviceToHost));

        const double forward_ms = std::chrono::duration<double, std::milli>(forward_end - forward_start).count();
        const double total_ms = std::chrono::duration<double, std::milli>(total_end - total_start).count();
        write_grid_csv("results/cuda_velocity_model.csv", velocity);
        write_grid_csv("results/cuda_rtm_image.csv", host_image);
        write_grid_csv("results/cuda_rtm_image_laplacian.csv", host_image_lap);
        write_traces_csv("results/cuda_receiver_gather.csv", host_traces);
        std::cout << "CUDA 3D acoustic RTM prototype completed\n"
                  << "GPU: " << prop.name << "\nGrid: " << NX << 'x' << NY << 'x' << NZ
                  << ", time steps: " << NT << "\n"
                  << "Forward modeling: " << forward_ms << " ms\n"
                  << "Forward + RTM imaging: " << total_ms << " ms\n"
                  << "Wrote results/cuda_rtm_image.csv, results/cuda_rtm_image_laplacian.csv, and results/cuda_receiver_gather.csv\n";

        CUDA_CHECK(cudaFree(d_velocity)); CUDA_CHECK(cudaFree(d_damping));
        CUDA_CHECK(cudaFree(d_prev)); CUDA_CHECK(cudaFree(d_curr));
        CUDA_CHECK(cudaFree(d_next)); CUDA_CHECK(cudaFree(d_snapshots));
        CUDA_CHECK(cudaFree(d_traces)); CUDA_CHECK(cudaFree(d_image));
        CUDA_CHECK(cudaFree(d_image_lap));
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }
    return 0;
}