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

inline std::size_t index3(int x, int y, int z) {
    return (static_cast<std::size_t>(z) * NY + y) * NX + x;
}

float ricker(int t) {
    const float time = t * DT;
    const float a = PI * F0 * (time - 1.5f / F0);
    const float a2 = a * a;
    return (1.0f - 2.0f * a2) * std::exp(-a2);
}

void build_velocity(std::vector<float>& velocity) {
    for (int z = 0; z < NZ; ++z) {
        for (int y = 0; y < NY; ++y) {
            for (int x = 0; x < NX; ++x) {
                // A layer boundary steps across x to mimic a simple fault-like offset.
                const int offset = (x >= NX / 2) ? 4 : 0;
                float v = z < 20 + offset ? 1800.0f
                        : z < 40 + offset ? 2300.0f : 2800.0f;
                const int dx = x - 43, dy = y - 35, dz = z - 48;
                if (dx * dx + dy * dy + dz * dz < 7 * 7) v = 3200.0f;
                velocity[index3(x, y, z)] = v;
            }
        }
    }
}

void step_wave(const std::vector<float>& previous,
               const std::vector<float>& current,
               std::vector<float>& next,
               const std::vector<float>& velocity,
               float source_amplitude) {
    const float inv_dx2 = 1.0f / (DX * DX);
    for (int z = 0; z < NZ; ++z) {
        for (int y = 0; y < NY; ++y) {
            for (int x = 0; x < NX; ++x) {
                const std::size_t i = index3(x, y, z);
                if (x == 0 || y == 0 || z == 0 || x == NX - 1 || y == NY - 1 || z == NZ - 1) {
                    next[i] = 0.0f;
                    continue;
                }
                const float c = current[i];
                const float lap = (current[index3(x - 1, y, z)] + current[index3(x + 1, y, z)]
                                 + current[index3(x, y - 1, z)] + current[index3(x, y + 1, z)]
                                 + current[index3(x, y, z - 1)] + current[index3(x, y, z + 1)]
                                 - 6.0f * c) * inv_dx2;
                next[i] = 2.0f * c - previous[i]
                        + velocity[i] * velocity[i] * DT * DT * lap;
                if (x == SX && y == SY && z == SZ) next[i] += source_amplitude;
            }
        }
    }
}

void inject_receivers(std::vector<float>& field,
                      const std::vector<float>& traces, int t) {
    for (int r = 0; r < NR; ++r) {
        const int x = 4 + r;
        field[index3(x, NY / 2, SZ)] += traces[static_cast<std::size_t>(t) * NR + r];
    }
}

void write_grid_csv(const std::string& path, const std::vector<float>& grid) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("Cannot open output file: " + path);
    out << "x,y,z,value\n" << std::setprecision(7);
    for (int z = 0; z < NZ; ++z)
        for (int y = 0; y < NY; ++y)
            for (int x = 0; x < NX; ++x)
                out << x << ',' << y << ',' << z << ',' << grid[index3(x, y, z)] << '\n';
}

void write_traces_csv(const std::string& path, const std::vector<float>& traces) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("Cannot open output file: " + path);
    out << "time_step,time_seconds,receiver,amplitude\n" << std::setprecision(7);
    for (int t = 0; t < NT; ++t)
        for (int r = 0; r < NR; ++r)
            out << t << ',' << t * DT << ',' << r << ','
                << traces[static_cast<std::size_t>(t) * NR + r] << '\n';
}
}

int main() {
    try {
        std::filesystem::create_directories("results");
        std::vector<float> velocity(CELLS), previous(CELLS, 0.0f),
                           current(CELLS, 0.0f), next(CELLS, 0.0f);
        std::vector<float> snapshots(SNAPSHOT_VALUES, 0.0f);
        std::vector<float> traces(static_cast<std::size_t>(NT) * NR, 0.0f);
        build_velocity(velocity);

        const auto start = std::chrono::steady_clock::now();
        for (int t = 0; t < NT; ++t) {
            step_wave(previous, current, next, velocity, ricker(t));
            std::copy(next.begin(), next.end(), snapshots.begin() + static_cast<std::size_t>(t) * CELLS);
            for (int r = 0; r < NR; ++r) {
                const int x = 4 + r;
                traces[static_cast<std::size_t>(t) * NR + r] = next[index3(x, NY / 2, SZ)];
            }
            previous.swap(current);
            current.swap(next);
        }
        const auto forward_end = std::chrono::steady_clock::now();

        std::fill(previous.begin(), previous.end(), 0.0f);
        std::fill(current.begin(), current.end(), 0.0f);
        std::fill(next.begin(), next.end(), 0.0f);
        std::vector<float> image(CELLS, 0.0f);
        for (int t = NT - 1; t >= 0; --t) {
            step_wave(previous, current, next, velocity, 0.0f);
            inject_receivers(next, traces, t);
            const std::size_t offset = static_cast<std::size_t>(t) * CELLS;
            for (std::size_t i = 0; i < CELLS; ++i)
                image[i] += snapshots[offset + i] * next[i];
            previous.swap(current);
            current.swap(next);
        }
        const auto end = std::chrono::steady_clock::now();
        const double forward_ms = std::chrono::duration<double, std::milli>(forward_end - start).count();
        const double total_ms = std::chrono::duration<double, std::milli>(end - start).count();

        write_grid_csv("results/cpu_velocity_model.csv", velocity);
        write_grid_csv("results/cpu_rtm_image.csv", image);
        write_traces_csv("results/cpu_receiver_gather.csv", traces);
        std::cout << "CPU 3D acoustic RTM prototype completed\n"
                  << "Grid: " << NX << 'x' << NY << 'x' << NZ << ", time steps: " << NT << '\n'
                  << "Forward modeling: " << forward_ms << " ms\n"
                  << "Forward + RTM imaging: " << total_ms << " ms\n"
                  << "Wrote results/cpu_rtm_image.csv and results/cpu_receiver_gather.csv\n";
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }
    return 0;
}