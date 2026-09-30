# Validation scope

The package manifest identifies the distributed binary. Validation used an
NVIDIA H100 PCIe (114 SM, compute capability 9.0):

- 12 CPU contract tests and the public C/C++ examples passed.
- 324 numerical checks and 324 caller graphs with 648 replays passed.
- Two memcheck suites reported zero errors.
- API checks cover affine calls, replacement values, ownership, invalid inputs,
  failure cleanup and device discovery outside Plan preparation.

Initial timing qualification used qap15 A, L2CTA3D AT and neos-3025225_lp AT.
The [full benchmark](../benchmarks/README.md#results) covers 200 directions and
both scalar modes. All 44,496 vector checks passed finite-value, repeatability
and backward-error criteria. Twelve SuiteSparse directions per mode failed the
forward-error threshold for all four methods; cuSPARSE ALG2 has one additional
failing case. See [numerical criteria and results](API.md#numerical-checks).

Other GPU targets have compilation and CPU identity/resource test coverage;
runtime validation is limited to H100 PCIe. Ordinary FP64 execution and
preparation-time probes do not guarantee an error bound for every future input.
