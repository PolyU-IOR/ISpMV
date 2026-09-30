#include <ispmv/ispmv.h>
#include <cuda_runtime_api.h>
#include <math.h>
#include <stdio.h>

#define CUDA_TRY(call) do { \
    cudaError_t error = (call); \
    if (error != cudaSuccess) { \
        fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(error)); \
        goto cleanup; \
    } \
} while (0)

int main(void) {
    /* A = [[1, 0, 2], [0, 0, 0]], x = [1, 2, 3]. */
    const int32_t row[] = {0, 2, 2}, col[] = {0, 2};
    const double val[] = {1, 2}, x[] = {1, 2, 3};
    double y[] = {10, 20};
    int32_t *d_row = NULL, *d_col = NULL;
    double* d_val = NULL;
    ispmv_plan* plan = NULL;
    ispmv_device* device = NULL;
    ispmv_csr matrix;
    ispmv_config options;
    int result = 1;
    ispmv_status status = ispmv_device_create(-1, &device);
    if (status != ISPMV_SUCCESS) goto sdk_error;
    CUDA_TRY(cudaMalloc((void**)&d_row, sizeof(row)));
    CUDA_TRY(cudaMalloc((void**)&d_col, sizeof(col)));
    CUDA_TRY(cudaMalloc((void**)&d_val, sizeof(val)));
    CUDA_TRY(cudaMemcpy(d_row, row, sizeof(row), cudaMemcpyHostToDevice));
    CUDA_TRY(cudaMemcpy(d_col, col, sizeof(col), cudaMemcpyHostToDevice));
    CUDA_TRY(cudaMemcpy(d_val, val, sizeof(val), cudaMemcpyHostToDevice));
    CUDA_TRY(cudaDeviceSynchronize());
    ispmv_csr_init(&matrix);
    ispmv_config_init(&options);
    matrix.rows = 2; matrix.columns = 3; matrix.nonzeros = 2;
    matrix.device_row_offsets = d_row; matrix.device_column_indices = d_col;
    matrix.device_values = d_val;
    options.copy_device_matrix = 0;
    status = ispmv_prepare(device, &matrix, &options, &plan);
    if (status != ISPMV_SUCCESS) goto sdk_error;
    /* Host vectors are a convenience; the plan above was built from device CSR. */
    status = ispmv_apply_host(plan, 1.0, x, 0.0, y);
    if (status != ISPMV_SUCCESS) goto sdk_error;
    result = !(fabs(y[0] - 7.0) < 1e-12 && fabs(y[1]) < 1e-12);
    if (!result) printf("[%g,%g]\n", y[0], y[1]);
    else fprintf(stderr, "ISpMV: Ax validation failed\n");
    goto cleanup;
sdk_error:
    fprintf(stderr, "ISpMV: %s: %s\n", ispmv_status_message(status), ispmv_error_message());
cleanup:
    cudaDeviceSynchronize();
    ispmv_release(plan);
    ispmv_device_release(device);
    cudaFree(d_val); cudaFree(d_col); cudaFree(d_row);
    return result;
}
