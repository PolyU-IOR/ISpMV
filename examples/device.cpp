#include <ispmv/ispmv.hpp>
#include <cuda_runtime_api.h>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <stdexcept>

static void cuda_check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
struct Buffers {
    double *x = nullptr, *y = nullptr;
    int32_t *rows = nullptr, *columns = nullptr;
    double* values = nullptr;
    cudaStream_t stream = nullptr;
    ~Buffers() {
        if (stream) cudaStreamSynchronize(stream);
        cudaFree(x); cudaFree(y);
        cudaFree(rows); cudaFree(columns); cudaFree(values);
        if (stream) cudaStreamDestroy(stream);
    }
};
struct Finish {
    cudaStream_t stream;
    ~Finish() { cudaStreamSynchronize(stream); }
};

int main() {
    try {
        // A = [[1, 0, 2], [0, 0, 0]], so A*x = [7,0].
        const int32_t row[] = {0, 2, 2}, col[] = {0, 2};
        const double val[] = {1, 2}, x[] = {1, 2, 3};
        double y[] = {10, 20};
        ispmv::Device hardware;  // once, outside plan timing
        Buffers device;
        cuda_check(cudaStreamCreate(&device.stream));
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&device.x), sizeof(x)));
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&device.y), sizeof(y)));
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&device.rows), sizeof(row)));
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&device.columns), sizeof(col)));
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&device.values), sizeof(val)));
        cuda_check(cudaMemcpy(device.x, x, sizeof(x), cudaMemcpyHostToDevice));
        cuda_check(cudaMemcpy(device.y, y, sizeof(y), cudaMemcpyHostToDevice));
        cuda_check(cudaMemcpy(device.rows, row, sizeof(row), cudaMemcpyHostToDevice));
        cuda_check(cudaMemcpy(device.columns, col, sizeof(col), cudaMemcpyHostToDevice));
        cuda_check(cudaMemcpy(device.values, val, sizeof(val), cudaMemcpyHostToDevice));
        cuda_check(cudaDeviceSynchronize());
        // A solver may pass its existing device CSR directly, after presolve/scaling.
        auto config = ispmv::default_config(); config.copy_device_matrix = 0;
        ispmv::Plan plan(hardware, ispmv::device_csr_view(2, 3, 2,
            device.rows, device.columns, device.values), config);
        Finish finish{device.stream};  // finish queued work before plan destruction
        for (int iteration = 0; iteration < 100; ++iteration) {
            // In a solver, update device.x on this stream before each call.
            plan.spmv(device.x, device.y, device.stream);
        }
        cuda_check(cudaStreamSynchronize(device.stream));
        cuda_check(cudaMemcpy(y, device.y, sizeof(y), cudaMemcpyDeviceToHost));
        if (!(std::abs(y[0] - 7.0) < 1e-12 && std::abs(y[1]) < 1e-12))
            throw std::runtime_error("Ax validation failed");
        std::cout << "[" << y[0] << "," << y[1] << "]\n";
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "ISpMV: %s\n", error.what());
        return 1;
    }
}
