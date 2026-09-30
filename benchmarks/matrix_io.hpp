#pragma once
#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <inttypes.h>
#include <vector>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        const cudaError_t status_ = (call);                                      \
        if (status_ != cudaSuccess) {                                            \
            std::fprintf(stderr, "CUDA failure at %s:%d: %s\n", __FILE__,      \
                         __LINE__, cudaGetErrorString(status_));                  \
            std::exit(2);                                                        \
        }                                                                       \
    } while (0)

struct HostCsr {
    int32_t rows = 0;
    int32_t cols = 0;
    int64_t nnz = 0;
    std::vector<int32_t> row_offsets;
    std::vector<int32_t> column_indices;
    std::vector<double> values;
};

static uint32_t read_u32_le(const unsigned char* p) {
    return static_cast<uint32_t>(p[0]) |
           (static_cast<uint32_t>(p[1]) << 8) |
           (static_cast<uint32_t>(p[2]) << 16) |
           (static_cast<uint32_t>(p[3]) << 24);
}

static uint64_t read_u64_le(const unsigned char* p) {
    return static_cast<uint64_t>(read_u32_le(p)) |
           (static_cast<uint64_t>(read_u32_le(p + 4)) << 32);
}

static bool read_exact(FILE* file, void* data, size_t bytes) {
    return bytes == 0 || std::fread(data, 1, bytes, file) == bytes;
}

static uint64_t file_size(FILE* file) {
    if (std::fseek(file, 0, SEEK_END) != 0) return 0;
    const long end = std::ftell(file);
    if (end < 0 || std::fseek(file, 0, SEEK_SET) != 0) return 0;
    return static_cast<uint64_t>(end);
}

static bool load_csr(const char* path, HostCsr& matrix) {
    static const unsigned char csrbin_magic[8] =
        {'C', 'S', 'R', 'B', 'I', 'N', '1', 0};
    static const unsigned char hprcsr_magic[8] =
        {'H', 'P', 'R', 'C', 'S', 'R', '1', 0};
    unsigned char header[72] = {};
    FILE* file = std::fopen(path, "rb");
    if (!file) {
        std::perror(path);
        return false;
    }
    const uint64_t bytes = file_size(file);
    if (!bytes || !read_exact(file, header, 8)) {
        std::fprintf(stderr, "truncated CSR header: %s\n", path);
        std::fclose(file);
        return false;
    }
    const bool hprcsr = std::memcmp(header, hprcsr_magic, 8) == 0;
    const bool csrbin = std::memcmp(header, csrbin_magic, 8) == 0;
    const size_t header_bytes = hprcsr ? sizeof(header) : 28u;
    if ((!hprcsr && !csrbin) ||
        !read_exact(file, header + 8, header_bytes - 8)) {
        std::fprintf(stderr, "unsupported CSR file: %s\n", path);
        std::fclose(file);
        return false;
    }

    const uint32_t version = read_u32_le(header + 8);
    const uint64_t rows64 = hprcsr ? read_u64_le(header + 24)
                                   : read_u32_le(header + 12);
    const uint64_t cols64 = hprcsr ? read_u64_le(header + 32)
                                   : read_u32_le(header + 16);
    const uint64_t nnz64 = hprcsr ? read_u64_le(header + 40)
                                  : read_u64_le(header + 20);
    if (version != 1 || rows64 == 0 || cols64 == 0 ||
        rows64 > INT32_MAX || cols64 > INT32_MAX || nnz64 > INT32_MAX) {
        std::fprintf(stderr, "unsupported CSR dimensions/version: %s\n", path);
        std::fclose(file);
        return false;
    }
    if (hprcsr &&
        (read_u32_le(header + 12) != sizeof(header) ||
         read_u32_le(header + 16) != 32 ||
         read_u32_le(header + 20) != 64 ||
         read_u64_le(header + 48) != rows64 + 1 ||
         read_u64_le(header + 56) != nnz64 ||
         read_u64_le(header + 64) != nnz64)) {
        std::fprintf(stderr, "unsupported HPRCSR1 header: %s\n", path);
        std::fclose(file);
        return false;
    }

    const uint64_t canonical_bytes = static_cast<uint64_t>(header_bytes) +
        4u * (rows64 + 1u) + 12u * nnz64;
    const uint64_t compact_bytes = 28u + 4u * rows64 + 8u * nnz64;
    const bool canonical = bytes == canonical_bytes;
    const bool compact = csrbin && bytes == compact_bytes;
    if (!canonical && !compact) {
        std::fprintf(stderr, "unknown CSR layout/size: %s\n", path);
        std::fclose(file);
        return false;
    }

    matrix.rows = static_cast<int32_t>(rows64);
    matrix.cols = static_cast<int32_t>(cols64);
    matrix.nnz = static_cast<int64_t>(nnz64);
    matrix.row_offsets.resize(static_cast<size_t>(rows64) + 1);
    matrix.column_indices.resize(static_cast<size_t>(nnz64));
    matrix.values.resize(static_cast<size_t>(nnz64));
    bool ok = true;
    if (canonical) {
        ok &= read_exact(file, matrix.row_offsets.data(),
                         (static_cast<size_t>(rows64) + 1) * sizeof(int32_t));
    } else {
        matrix.row_offsets[0] = 0;
        ok &= read_exact(file, matrix.row_offsets.data() + 1,
                         static_cast<size_t>(rows64) * sizeof(int32_t));
    }
    ok &= read_exact(file, matrix.column_indices.data(),
                     static_cast<size_t>(nnz64) * sizeof(int32_t));
    if (canonical) {
        ok &= read_exact(file, matrix.values.data(),
                         static_cast<size_t>(nnz64) * sizeof(double));
    } else {
        std::vector<float> fp32_values(static_cast<size_t>(nnz64));
        ok &= read_exact(file, fp32_values.data(),
                         static_cast<size_t>(nnz64) * sizeof(float));
        if (ok) {
            for (size_t i = 0; i < fp32_values.size(); ++i) {
                matrix.values[i] = fp32_values[i];
            }
        }
    }
    std::fclose(file);
    if (!ok || matrix.row_offsets.front() != 0 ||
        matrix.row_offsets.back() != matrix.nnz) {
        std::fprintf(stderr, "invalid or truncated CSR arrays: %s\n", path);
        return false;
    }
    for (int32_t row = 0; row < matrix.rows; ++row) {
        if (matrix.row_offsets[row] > matrix.row_offsets[row + 1]) return false;
    }
    for (int64_t p = 0; p < matrix.nnz; ++p) {
        if (matrix.column_indices[p] < 0 ||
            matrix.column_indices[p] >= matrix.cols ||
            !std::isfinite(matrix.values[p])) return false;
    }
    std::fprintf(stderr,
                 "loaded %s rows=%d cols=%d nnz=%" PRId64 " layout=%s\n",
                 path, matrix.rows, matrix.cols, matrix.nnz,
                 hprcsr ? "HPRCSR1/FP64" :
                 (canonical ? "CSRBIN1/FP64" : "CSRBIN1/FP32"));
    return true;
}
