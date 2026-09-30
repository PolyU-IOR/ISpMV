# Installation

Download the SDK, verify the installation, then run a benchmark or link ISpMV
into your application. Run the commands below in one Bash session, starting in
a fresh working directory.

## Requirements

| Component | Requirement |
|---|---|
| Platform | Linux x86-64; compiled targets sm_80/86/89/90/100/120; runtime qualification on H100 PCIe |
| NVIDIA libraries | CUDA Runtime 13 and cuBLAS 13; installed separately with a compatible driver |
| System libraries | glibc ≥ 2.34; GLIBCXX_3.4.29 and CXXABI_1.3.9 |
| Build tools | CMake ≥ 3.24 and a C11/C++17 compiler |
| Examples | CUDA Runtime development headers/libraries for device CSR allocation |
| Benchmark | CUDA 13.3 development files including cuSPARSE SpMVOp, OpenMP and C++17 |
| Archive tools | tar and sha256sum |

The SDK has no PTX fallback. See [hardware support](HARDWARE.md) and the package
manifest for compiled targets and actual runtime qualification. Public headers
can be parsed without CUDA headers; CUDA allocation/stream calls in client
applications require CUDA development files. No CUDA compiler is required to
build the included C/C++ clients. The runtime API does not require Python or cuSPARSE.

## Download

Download the named binary SDK and its `.sha256` companion from the
[release](https://github.com/PolyU-IOR/ISpMV/releases/tag/v0.1.0), using an account
with repository access when required. In the download directory:

```sh
export ISPMV_PACKAGE=ISpMV-0.1.0-linux-x86_64-cuda13
sha256sum -c "$ISPMV_PACKAGE.tar.gz.sha256" && tar -xzf "$ISPMV_PACKAGE.tar.gz"
export ISPMV_SDK="$PWD/$ISPMV_PACKAGE"
(cd "$ISPMV_SDK" && sha256sum -c SHA256SUMS)
```

The checksum must report `OK` before extraction. Git clones and GitHub's
automatic Source code archives do not contain the library. The SDK includes
`lib/libispmv.so.1`, headers, examples and CMake package files.

## Setup

Set `CUDA_ROOT` to your CUDA installation:

```sh
export CUDA_ROOT=/usr/local/cuda-13.3
export LD_LIBRARY_PATH="$ISPMV_SDK/lib:$CUDA_ROOT/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
nvidia-smi
ldd "$ISPMV_SDK/lib/libispmv.so.1"
```

Every dependency must resolve; fix any `not found` entry before running examples.
Use a driver compatible with your CUDA libraries. Do not use CUDA `stubs/` as
runtime libraries. The package is relocatable and contains no private search paths.

## Verify the installation

Build and run the device-buffer example, which constructs the plan on GPU:

```sh
cmake -S "$ISPMV_SDK/examples" -B example-build \
  -DCMAKE_PREFIX_PATH="$ISPMV_SDK" -DCUDAToolkit_ROOT="$CUDA_ROOT" \
  -DISPMV_EXAMPLE_DEVICE=ON
cmake --build example-build
./example-build/ispmv_device
```

The example prints `[7,0]` after repeated SpMV calls. To try synchronous
host-vector calls with a GPU-built plan:

```sh
./example-build/ispmv_host_c
./example-build/ispmv_host_cpp
```

Both print `[7,0]`. All three examples upload the small fixture's CSR,
prepare using device CSR, execute on GPU and return zero on success. In a solver
with CSR already on GPU, use those existing buffers directly. Building clients
or finding the CMake package does not run GPU work.

## GPU coverage

The package contains sm_80, sm_86, sm_89, sm_90, sm_100 and sm_120 device code.
An explicit device handle discovers identity and resources once, before Plan
timing, and supplies immutable hardware information to preparation. Resource
adaptation uses the actual GPU configuration; H100 PCIe and SXM are distinguished
when reported by the device. Runtime qualification currently covers H100 PCIe;
other compiled targets are not claims of completed device testing. See
[hardware adaptation and limitations](HARDWARE.md).

## File input

ISpMV operates on CSR arrays. If your application already has device CSR after
presolve/scaling, skip loading and upload and proceed to application integration.

The benchmark reads A and Aᵀ in the project's custom CSRBIN1 or HPRCSR1 binary
CSR formats (zero-based) and uploads them before timing. For MTX, MPS or MAT input, convert to CSR in your
application or data preparation tools first. Readers for those formats and
raw matrix datasets are not bundled. See [input layout](../benchmarks/README.md#input).
File I/O, conversion and upload are outside GPU plan timing.

## Run a benchmark

```sh
cmake -S "$ISPMV_SDK/benchmarks" -B benchmark-build \
  -DCMAKE_PREFIX_PATH="$ISPMV_SDK" -DCUDAToolkit_ROOT="$CUDA_ROOT" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build benchmark-build
./benchmark-build/ispmv_benchmark LABEL A.csrbin AT.csrbin ispmv
```

Replace the input filenames with your own CSR files. Select `cusparse_alg2`,
`spmvop_alg1` or `spmvop_alg2` as the final argument for a comparison method.
Each invocation measures both orientations and both scalar modes (1,0) and
(1.25,-0.375). It prints timing and numerical checks using generated vectors.
To compute with your own vectors, use the API in your application.
See [benchmark protocol and output](../benchmarks/README.md).

## Application integration

Start a standalone C++ application from the complete device-buffer example:

```sh
mkdir my-app
cp "$ISPMV_SDK/examples/device.cpp" my-app/main.cpp
cat > my-app/CMakeLists.txt <<'EOF'
cmake_minimum_required(VERSION 3.24)
project(MyApp LANGUAGES CXX)
find_package(ISpMV 0.1.0 CONFIG REQUIRED COMPONENTS runtime)
find_package(CUDAToolkit REQUIRED)
add_executable(my_app main.cpp)
target_compile_features(my_app PRIVATE cxx_std_17)
target_link_libraries(my_app PRIVATE ISpMV::runtime CUDA::cudart)
EOF
cmake -S my-app -B my-app/build \
  -DCMAKE_PREFIX_PATH="$ISPMV_SDK" -DCUDAToolkit_ROOT="$CUDA_ROOT"
cmake --build my-app/build
./my-app/build/my_app
```

Edit `my-app/main.cpp` to supply your matrix and vectors, replacing the example's
fixed arrays, dimensions and result checks. Follow this lifecycle:

1. Initialize `ispmv::Device` once for the current CUDA device, outside Plan timing.
2. Obtain device CSR with `int32_t` offsets/indices and `double` values after
   presolve/scaling. Complete producing GPU work. See [input requirements](API.md#inputs-and-ownership).
3. Create `device_csr_view`, then `Plan(device, csr, config)` once for the fixed
   matrix. With `copy_device_matrix=0`, keep CSR immutable and alive through use.
4. Allocate x with `cols` entries and y with `rows` entries. Initialize y when
   beta is nonzero. Update x on the same stream and call `plan.spmv(...)`.
5. Synchronize before reading results on CPU or releasing plans and buffers.
   Keep intermediate vectors on GPU when subsequent operations use them there.

Reuse the plan when vectors or scalars change. For A and Aᵀ, construct two
independent plans using the same device handle. The example includes CUDA error
handling and synchronization during cleanup; retain those when adapting it.

The SDK's CMake package is installed in `lib/cmake/ISpMV/`.
See [API](API.md) for ownership, streams and numerical behavior.
