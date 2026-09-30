# Hardware support

Initialize `ispmv::Device` (`ispmv_device_create` in C) once for the current CUDA
device and reuse it when preparing plans. Discovery is independent of the matrix
and is measured separately from Plan construction.

The immutable handle reports device identity, model/board variant, compute
capability, SM count, memory, cache and launch limits. Plan adaptation uses the
reported resources, including reduced configurations. H100 PCIe/SXM and A100
memory variants are distinguished when reported by the device; ambiguous board
names return `unknown` while resource-based adaptation remains available.

The package contains sm_80, sm_86, sm_89, sm_90, sm_100 and sm_120 device code,
covering A100, H100/H200, B200, RTX 4090 and RTX 5090 architectures. There is no
PTX fallback; discovery returns UNSUPPORTED for absent targets. Runtime
validation covers H100 PCIe. See the [validation scope](VALIDATION.md) and
package manifest for the tested configuration.

Device information describes capabilities, not current free memory or measured
clocks. Recreate the handle after device reset or reconfiguration. See the
[API](API.md#lifecycle) for handle ownership and [benchmark protocol](../benchmarks/README.md#protocol)
for timing.

References: [CUDA compute capability](https://developer.nvidia.com/cuda/gpus),
[NVIDIA H100 specifications](https://www.nvidia.com/en-us/data-center/h100/).
