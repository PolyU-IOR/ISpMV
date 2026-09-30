#include <ispmv/ispmv.hpp>
#include <cuda_runtime_api.h>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <stdexcept>

static void cuda_check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
struct MatrixBuffers {
    int32_t *rows = nullptr, *columns = nullptr;
    double* values = nullptr;
    ~MatrixBuffers() {
        cudaDeviceSynchronize();
        cudaFree(values); cudaFree(columns); cudaFree(rows);
    }
};

struct Finish { ~Finish() { cudaDeviceSynchronize(); } };

int main() {
    try {
        const int32_t row[] = {0, 2, 2}, col[] = {0, 2};
        const double val[] = {1, 2}, x[] = {1, 2, 3};
        double y[] = {10, 20};
        ispmv::Device hardware;
        MatrixBuffers matrix;
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&matrix.rows), sizeof(row)));
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&matrix.columns), sizeof(col)));
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&matrix.values), sizeof(val)));
        cuda_check(cudaMemcpy(matrix.rows, row, sizeof(row), cudaMemcpyHostToDevice));
        cuda_check(cudaMemcpy(matrix.columns, col, sizeof(col), cudaMemcpyHostToDevice));
        cuda_check(cudaMemcpy(matrix.values, val, sizeof(val), cudaMemcpyHostToDevice));
        cuda_check(cudaDeviceSynchronize());
        auto config = ispmv::default_config(); config.copy_device_matrix = 0;
        ispmv::Plan plan(hardware, ispmv::device_csr_view(2, 3, 2,
            matrix.rows, matrix.columns, matrix.values), config);
        Finish finish;  // complete any submitted work before plan destruction
        // Synchronous host-vector convenience call on a GPU-built plan.
        plan.apply_host(1.0, x, 0.0, y);
        if (!(std::abs(y[0] - 7.0) < 1e-12 && std::abs(y[1]) < 1e-12))
            throw std::runtime_error("Ax validation failed");
        std::cout << "[" << y[0] << "," << y[1] << "]\n";
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "ISpMV: %s\n", error.what());
        return 1;
    }
}
