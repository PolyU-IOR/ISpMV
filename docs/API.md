# C/C++ API reference

## Lifecycle

Initialize a device handle once, prepare a plan from device-resident CSR on the
GPU, reuse it with device vectors, then synchronize before releasing resources.

| Operation | C | C++ |
|---|---|---|
| Discover the current GPU | `ispmv_device_create` | `Device` |
| Query hardware | `ispmv_device_query` | `Device::info` |
| Initialize inputs | `ispmv_csr_init`, `ispmv_config_init` | `device_csr_view`, `default_config` |
| Prepare on GPU | `ispmv_prepare` with device CSR | `Plan(device, A)` |
| Device SpMV | `ispmv_apply` | `Plan::spmv` |
| Replace values for one call | `ispmv_apply_with_values` | `Plan::apply_with_values` |
| Synchronous host vectors | `ispmv_apply_host` | `Plan::apply_host` |
| Query shape and ownership | `ispmv_query` | `Plan::info` |
| Release plan / device handle | `ispmv_release` / `ispmv_device_release` | Move-only RAII |

C calls return `ispmv_status`. On failure, read `ispmv_error_message()` immediately
on the same host thread; the next status-returning call replaces it. C++ throws
`ispmv::Error`. Also check asynchronous errors when synchronizing CUDA streams.

Initialize the immutable device handle outside Plan timing. `ordinal=-1` selects
the current device; an explicit ordinal must match it. Discovery does not change
the current GPU or perform matrix-dependent work. The handle may be shared by
concurrent preparations on host threads using that device. Complete preparation
before releasing it; existing plans remain valid. Recreate the handle after
device reset or reconfiguration. See [hardware adaptation](HARDWARE.md).

## Inputs and ownership

The C/C++ API accepts CSR arrays directly. Row offsets have `rows + 1` entries;
column indices and values each have `nonzeros` entries. For index base `b`,
the first and last offsets are `b` and `nonzeros + b`. File input must be
converted to CSR and uploaded before GPU plan preparation.

- Use finite FP64 values and scalars, signed 32-bit CSR indices, and zero- or
  one-based indexing. Initialize descriptors and preserve `struct_size`; set it
  before querying information structures.
- Dimensions must be positive; empty rows and zero nonzeros are allowed.
  Unsorted columns and duplicate entries are supported. Offsets, column bounds
  and values are validated; nnz must fit the selected index base.
- Supply device CSR using `device_row_offsets`, `device_column_indices` and
  `device_values`, or C++ `device_csr_view`. Device fields are authoritative if
  any is supplied; host fields are then ignored. Complete input-producing GPU
  work before synchronous preparation, outside graph capture.
- Device CSR is analyzed on GPU; only scalar status/statistics return to CPU.
  Copy mode (`copy_device_matrix=1`, default) owns D2D copies. With zero,
  zero-based arrays are borrowed and must stay immutable and alive until plan
  release. One-based input always owns normalized copies. Query actual ownership
  with `ispmv_query`.
- Plans and vectors belong to the current CUDA device. Vectors must have
  sufficient lengths. `beta == 0` does not read y; `alpha == 0` permits null x.
  For nonzero alpha, x and y must not overlap. Scalars are finite host values.

For first-order solvers, provide the CSR produced after GPU presolve and
row/column scaling. Plan wall time starts with completed device CSR and a ready
device handle and includes all matrix-dependent validation, allocation and
selection. Report device initialization and CSR upload separately.

Host-only CSR (`csr_view`) remains a compatibility input that uses CPU analysis.
For GPU preprocessing, upload CSR and use the device fields, as all included
examples do. `apply_host` refers to vector transfers and does not choose the
plan builder; it is synchronous and includes those transfers.

Reuse the plan when vectors or scalars change. `apply_with_values` accepts finite
device coefficients in the original CSR entry order, valid until completion;
it bypasses value-dependent representations and leaves the saved matrix unchanged.
Changes to sparsity, dimensions or device require a new plan.

## File input

Convert matrix files to CSR and upload before GPU preparation; the runtime API
does not read files. See the [installation guide](INSTALL.md#file-input) and
[benchmark input layout](../benchmarks/README.md#input).

## A and Aᵀ

Supply each orientation as device CSR and prepare it independently using the
same device handle. Both plans execute normally without a transpose flag.

```cpp
ispmv::Device device;
ispmv::Plan a_plan(device, a_device_csr);
ispmv::Plan at_plan(device, at_device_csr);
a_plan.spmv(alpha, x_device, beta, y_device, stream);
at_plan.spmv(alpha, u_device, beta, v_device, stream);
```

## Streams and graphs

Submit a plan from one host thread at a time. Same-stream calls may be queued;
complete outstanding work before switching streams or devices, releasing resources
or replacing the plan. Destruction and `ispmv_release` do not synchronize.
Independent plans may be ordered with CUDA streams and events.

Prepare outside graph capture. Device execution supports caller CUDA graphs;
captured pointers and scalar values are fixed. Recapture to change them. Keep
the plan and buffers alive until all launches finish and the graphs are destroyed.
Preparation and host-vector calls cannot be captured. First use or new buffer/scalar
patterns may require setup; allocation-free submission is not guaranteed.

## Numerical limits

Input coefficients are preserved exactly and execution uses ordinary FP64.
Reduction order, FMA and shared computation may differ from serial or vendor
results. `deterministic=1` (default) selects reproducible reductions for a fixed
plan, device and identical inputs; disabling it permits other reduction methods.
Cross-device or cross-build bitwise equality is not promised.

FP64 results can exceed a mixed forward-error tolerance despite small scaled
backward error. No extra precision compensation is applied. Preparation-time
probes do not guarantee bounds for all future vectors. FP32, complex values,
64-bit indices, device-pointer scalars and CPU execution are unsupported.

## Numerical checks

The benchmark uses nine validation vectors and a CPU `long double` reference;
the reference significand size is recorded in each result. This reference is
part of validation, outside GPU plan and SpMV timing. For computed output
$\hat y$, reference $z$, and
$s_i=|\alpha|\sum_j|M_{ij}x_j|+|\beta y_i^{(0)}|$, where $M$ is A or Aᵀ:

- Mixed forward error: $\max_i |\hat y_i-z_i|/\max(1,|z_i|) \le 10^{-11}$.
- Scaled backward error: $\max_i |\hat y_i-z_i|/\max(1,s_i) \le 10^{-13}$.
- Outputs and reference must be finite, and repeated runs with identical inputs
  must be bitwise identical. The extreme dynamic-range vector is exempt only
  from the forward-error threshold. All nine vectors must pass; timed outputs
  must also remain finite.

See [validation scope](VALIDATION.md) for the checks completed on this binary.
The [benchmark protocol](../benchmarks/README.md#protocol) reports numerical
acceptance separately from timing. A completed measurement does not itself
establish numerical acceptance or a universal error bound.

A case is one matrix, one orientation and one operation. Results from the full benchmark of the current released binary:

| Operation | Cases | All four methods fail | ISpMV fails |
|---|---:|---:|---:|
| SpMV(1, 0) | 200 | 12 | 12 |
| SpMV(α, β) | 200 | 12 | 12 |

The shared failures involve the same six SuiteSparse matrices in both orientations:
**24 of 400 cases** across the two operations. One additional SpMV(1, 0) case
fails only for cuSPARSE ALG2. All failures are forward-error threshold exceedances;
finite-value, bitwise-repeatability and backward-error checks pass for all methods.
These observations apply to the tested cases; see [numerical limits](#numerical-limits).

## Version compatibility

SDK 0.1.0 uses ABI 1, `libispmv.so.1` and symbol version `ISPMV_SDK_1`.
`ispmv_version()` returns the ABI major, not the SDK version. Use matching headers
and binaries. CMake accepts compatible same-major-and-minor SDK requests up to
the installed version and requires 64-bit clients. The SDK supports Linux only.
