#ifndef ISPMV_H
#define ISPMV_H
#include <ispmv/types.h>
#define ISPMV_ABI_VERSION 1u
#ifdef __cplusplus
extern "C" {
#endif

/* Initialize descriptors before assigning fields. GPU CSR is the preferred
 * preparation input. Arrays contain CSR32 indices and FP64 finite coefficients. */
typedef struct ispmv_csr {
    uint32_t struct_size;
    int32_t rows;
    int32_t columns;
    int64_t nonzeros;
    ispmv_index_base index_base;
    const int32_t* row_offsets;     /* host, rows+1 entries */
    const int32_t* column_indices;  /* host, nonzeros entries */
    const double* values;           /* host, nonzeros entries */
    /* Device arrays are authoritative if any is supplied; host fields are then
     * ignored and may be null. Supply all nonempty arrays on the current device.
     * Complete producing GPU work before preparation. Host-only input retains
     * the CPU builder for compatibility and comparison. */
    const int32_t* device_row_offsets;
    const int32_t* device_column_indices;
    const double* device_values;
} ispmv_csr;

/* Coefficients are preserved exactly; execution uses ordinary FP64. */
typedef struct ispmv_config {
    uint32_t struct_size;
    int deterministic;       /* default 1; fixed plan/device, identical inputs */
    int copy_device_matrix;  /* default 1; borrowed arrays must outlive plan */
} ispmv_config;
typedef struct ispmv_info {
    uint32_t struct_size;
    int32_t rows;
    int32_t columns;
    int64_t nonzeros;
    int deterministic;
    int owns_device_matrix;
} ispmv_info;

/* Read-only discovery output. This is descriptive, not an editable tuning API.
 * Unknown board variants retain their reported numeric capabilities. */
typedef struct ispmv_device_info {
    uint32_t struct_size;
    int32_t ordinal;
    char name[256], model[32], variant[16];
    unsigned char uuid[16];
    int32_t compute_major, compute_minor, multiprocessors;
    uint64_t memory_bytes;
    int32_t l2_bytes, shared_optin_bytes, persisting_l2_bytes, access_window_bytes;
    int32_t memory_bus_bits, core_clock_khz, memory_clock_khz, fp32_to_fp64_ratio;
    int32_t cluster_launch, pci_domain, pci_bus, pci_device;
} ispmv_device_info;

/* Discover the current device once, before plan timing. ordinal=-1 means
 * current; an explicit ordinal must match the application's current device.
 * The immutable handle can be shared by preparation calls on host threads
 * using that device. Recreate after device reset/reconfiguration. No matrix
 * work, kernel warmup or matrix-specific tuning is performed here. */
ISPMV_API ispmv_status ispmv_device_create(int ordinal, ispmv_device** output);
ISPMV_API ispmv_status ispmv_device_query(const ispmv_device* device, ispmv_device_info* info);
/* Finish concurrent preparation calls first. Existing plans remain valid. */
ISPMV_API void ispmv_device_release(ispmv_device* device);

ISPMV_API uint32_t ispmv_version(void); /* ABI major = ISPMV_ABI_VERSION */
ISPMV_API void ispmv_csr_init(ispmv_csr* matrix);
ISPMV_API void ispmv_config_init(ispmv_config* config);
/* Preparation on the current CUDA device. Output is null on failure.
 * Null config selects defaults. Device CSR is analyzed on GPU; only scalar
 * status/statistics return to CPU. Copy mode uses D2D copies. One-based input
 * always owns normalized copies. In borrowed mode arrays must remain immutable
 * and alive until release. Preparation is synchronous and outside graph capture.
 * Device plans use native generic or GPU-proven structural paths; host
 * structural detectors are not invoked. Host-only arrays may be released on return. */
ISPMV_API ispmv_status ispmv_prepare(
    const ispmv_device* device, const ispmv_csr* matrix,
    const ispmv_config* config, ispmv_plan** output);

/* Enqueue y = alpha*A*x + beta*y using device buffers and finite host scalars.
 * stream is cudaStream_t cast to void*. beta=0 never reads old y; alpha=0
 * permits null x. For nonzero alpha, x and y cannot overlap. Submit a plan from one host thread at a time. Same-stream calls may be
 * queued; complete earlier use before switching streams or releasing resources.
 * Retain buffers until completion. */
ISPMV_API ispmv_status ispmv_apply(ispmv_plan* plan,
    double alpha, const double* x, double beta, double* y, void* stream);
/* Replacement device coefficients in the original CSR entry order; must be
 * finite and remain alive through completion. The saved matrix is unchanged.
 * Changing sparsity requires preparing a new plan. */
ISPMV_API ispmv_status ispmv_apply_with_values(ispmv_plan* plan,
    const double* values, double alpha, const double* x, double beta,
    double* y, void* stream);
/* Synchronous convenience call, with host vectors and transfers. For repeated
 * GPU computation, keep vectors on the device and use ispmv_apply. */
ISPMV_API ispmv_status ispmv_apply_host(ispmv_plan* plan,
    double alpha, const double* x, double beta, double* y);
ISPMV_API ispmv_status ispmv_query(const ispmv_plan* plan, ispmv_info* info);
/* Complete pending work and release caller graphs that use the plan first.
 * Null is accepted. This function is not a stream synchronization API. */
ISPMV_API void ispmv_release(ispmv_plan* plan);
ISPMV_API const char* ispmv_status_message(ispmv_status status);
/* Immediate thread-local error until the next status-returning call.
 * Also check deferred device errors when synchronizing the CUDA stream. */
ISPMV_API const char* ispmv_error_message(void);
#ifdef __cplusplus
}
#endif
#endif
