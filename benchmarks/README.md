# Benchmarks

The benchmark compares ISpMV, cuSPARSE ALG2 and SpMVOp ALG1/ALG2 using
GPU-resident CSR. All methods use independent plans for A and Aᵀ.

## Build

Use the [binary SDK](../docs/INSTALL.md), CMake ≥ 3.24, a C++17 compiler,
OpenMP and CUDA 13.3 development files including the experimental SpMVOp API.
No CUDA compiler is required.

```sh
cmake -S "$ISPMV_SDK/benchmarks" -B benchmark-build \
  -DCMAKE_PREFIX_PATH="$ISPMV_SDK" -DCUDAToolkit_ROOT="$CUDA_ROOT" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build benchmark-build
```

## Run

```sh
./benchmark-build/ispmv_benchmark LABEL A.csrbin AT.csrbin ispmv
./benchmark-build/ispmv_benchmark LABEL A.csrbin AT.csrbin cusparse_alg2
./benchmark-build/ispmv_benchmark LABEL A.csrbin AT.csrbin spmvop_alg1
./benchmark-build/ispmv_benchmark LABEL A.csrbin AT.csrbin spmvop_alg2
```

Supply your own label and CSR files. Each invocation measures A and Aᵀ with
(α, β) = (1, 0) and (1.25, −0.375), and outputs preparation time, execution time
and numerical checks as a `COMPARISON_RESULT` JSON record. Numerical acceptance
is reported in each mode's `accepted` field.

## Input

Supply the matrix and its matching transpose as zero-based CSR32 with finite
FP64 values. The benchmark reads CSRBIN1 or HPRCSR1, custom binary CSR formats
used by this project. The SDK itself accepts CSR arrays directly.

<details>
<summary>Input format details</summary>

CSRBIN1/FP64 consists of `CSRBIN1\0`, uint32 version=1, uint32 rows/columns,
uint64 nnz, int32 row offsets (rows+1), int32 column indices (nnz) and FP64 values
(nnz), little endian. The reader also accepts its compact FP32-value layout and
converts values exactly to FP64 before upload. HPRCSR1/FP64 uses a 72-byte header:
8-byte `HPRCSR1\0`, four uint32 fields (version=1, header bytes=72, index bits=32,
value bits=64), then six uint64 fields (rows, columns, nnz, rows+1, nnz, nnz),
followed by the same CSR32/FP64 arrays.

</details>

## Protocol

Plan wall time starts from ready GPU-resident CSR and includes validation,
allocation and selection. A and Aᵀ are prepared separately, in that order;
ISpMV uses `copy_device_matrix=0`. File I/O, upload, vector allocation, vendor
handle creation and external numerical checks are excluded. Device discovery is
reported separately.

Execution uses 100 warmup calls, then three groups of 50 timed calls, each
preceded by ten untimed calls. Steady time is the median group average.
Published results take the median of three independent processes per direction,
then the geometric mean across each dataset. Speedups use unrounded means.

Numerical checks use nine vectors and CPU `long double` references outside
timing. See [criteria and results](../docs/API.md#numerical-checks).

## Results

**H100 PCIe 80GB · FP64 · CUDA 13.4.2 (Runtime 13.4.92) · cuSPARSE 12.8.6.72.**
SpMVOp ALG1/ALG2 use `cusparseSpMVOp`; cuSPARSE ALG2 uses `cusparseSpMV`.

Execution time in **μs** (geometric mean), excluding preprocessing (lower is better).
Parentheses show **ISpMV speedup** over each baseline.
**All 200 directions are included, including directions that failed the numerical criterion.**

| Dataset | Operation | ISpMV | SpMVOp ALG1 | SpMVOp ALG2 | cuSPARSE ALG2 |
|---|---|---:|---:|---:|---:|
| Mittelmann · 49 × 2 | SpMV(1, 0) | 20.25 | 33.36 (1.65×) | 31.48 (1.56×) | 43.85 (2.17×) |
|  | SpMV(α, β) | 22.25 | 35.70 (1.60×) | 33.63 (1.51×) | 49.47 (2.22×) |
| MIPLIB (nnz > 10M) · 18 × 2 | SpMV(1, 0) | 141.54 | 239.64 (1.69×) | 231.79 (1.64×) | 232.66 (1.64×) |
|  | SpMV(α, β) | 157.51 | 262.18 (1.66×) | 251.90 (1.60×) | 251.87 (1.60×) |
| SuiteSparse (nnz > 10M) · 33 × 2 | SpMV(1, 0) | 211.03 | 335.00 (1.59×) | 300.26 (1.42×) | 275.97 (1.31×) |
|  | SpMV(α, β) | 234.19 | 358.95 (1.53×) | 324.30 (1.38×) | 303.62 (1.30×) |

[Matrix list](matrices.csv): 100 matrices and 200 directions. Both scalar modes
are tested; MIPLIB and SuiteSparse include matrices with nnz > 10M.

See [preprocessing results](../README.md#benchmarks) and
[numerical criteria and results](../docs/API.md#numerical-checks).
In the overall figure, a win means a baseline/ISpMV execution-time ratio greater than 1.05.
