#pragma once
#include <ispmv/ispmv.h>
#include <memory>
#include <stdexcept>
#include <utility>

namespace ispmv {
using Csr = ispmv_csr;
using Config = ispmv_config;
using Info = ispmv_info;
inline Config default_config() {
    Config config; ispmv_config_init(&config); return config;
}
inline Csr csr_view(int32_t rows, int32_t columns, int64_t nonzeros,
    const int32_t* offsets, const int32_t* indices, const double* values) {
    Csr matrix; ispmv_csr_init(&matrix);
    matrix.rows = rows; matrix.columns = columns; matrix.nonzeros = nonzeros;
    matrix.row_offsets = offsets; matrix.column_indices = indices;
    matrix.values = values; return matrix;
}
inline Csr device_csr_view(int32_t rows, int32_t columns, int64_t nonzeros,
    const int32_t* offsets, const int32_t* indices, const double* values) {
    Csr matrix; ispmv_csr_init(&matrix);
    matrix.rows=rows; matrix.columns=columns; matrix.nonzeros=nonzeros;
    matrix.device_row_offsets=offsets; matrix.device_column_indices=indices;
    matrix.device_values=values; return matrix;
}
class Error : public std::runtime_error {
    ispmv_status status_;
public:
    explicit Error(ispmv_status status)
        : std::runtime_error(*ispmv_error_message() ? ispmv_error_message()
                                                    : ispmv_status_message(status)),
          status_(status) {}
    ispmv_status status() const noexcept { return status_; }
};
inline void check(ispmv_status status) {
    if (status != ISPMV_SUCCESS) throw Error(status);
}

class Device {
    struct Release { void operator()(ispmv_device* p) const noexcept { ispmv_device_release(p); } };
    std::unique_ptr<ispmv_device,Release> handle_;
public:
    explicit Device(int ordinal=-1) {
        ispmv_device* raw=nullptr;auto status=ispmv_device_create(ordinal,&raw);
        handle_.reset(raw);check(status);
    }
    Device(Device&&) noexcept=default;
    Device& operator=(Device&&) noexcept=default;
    Device(const Device&)=delete;
    Device& operator=(const Device&)=delete;
    ispmv_device* native_handle() const noexcept { return handle_.get(); }
    ispmv_device_info info() const {
        ispmv_device_info result{};result.struct_size=sizeof(result);
        check(ispmv_device_query(native_handle(),&result));return result;
    }
};

// Move-only owner. Complete GPU work and release caller graphs before reset,
// destruction, or move-assignment into an already populated destination.
class Plan {
    struct Release { void operator()(ispmv_plan* p) const noexcept { ispmv_release(p); } };
    std::unique_ptr<ispmv_plan, Release> handle_;
public:
    Plan() noexcept = default;
    explicit Plan(const Device& device, const Csr& matrix, const Config& config = default_config())
        : Plan(prepare(device, matrix, config)) {}
    Plan(Plan&&) noexcept = default;
    Plan& operator=(Plan&&) noexcept = default;
    Plan(const Plan&) = delete;
    Plan& operator=(const Plan&) = delete;
    static Plan adopt(ispmv_plan* owned) noexcept {
        Plan result; result.handle_.reset(owned); return result;
    }
    static Plan prepare(const Device& device, const Csr& matrix, const Config& config = default_config()) {
        ispmv_plan* raw = nullptr;
        const auto status = ispmv_prepare(device.native_handle(), &matrix, &config, &raw);
        auto result = adopt(raw); check(status); return result;
    }
    ispmv_plan* native_handle() const noexcept { return handle_.get(); }
    explicit operator bool() const noexcept { return bool(handle_); }
    void reset() noexcept { handle_.reset(); }
    ispmv_plan* release() noexcept { return handle_.release(); }
    // Device buffers; reuse the matrix analysis and prepared resources.
    void spmv(const double* x, double* y, void* stream = nullptr) {
        apply(1., x, 0., y, stream);
    }
    void spmv(double alpha, const double* x, double beta, double* y,
              void* stream = nullptr) {
        apply(alpha, x, beta, y, stream);
    }
    void apply(double alpha, const double* x, double beta, double* y,
               void* stream = nullptr) {
        check(ispmv_apply(native_handle(), alpha, x, beta, y, stream));
    }
    void apply_with_values(const double* values, double alpha, const double* x,
                           double beta, double* y, void* stream = nullptr) {
        check(ispmv_apply_with_values(native_handle(), values, alpha, x, beta, y, stream));
    }
    void apply_host(double alpha, const double* x, double beta, double* y) {
        check(ispmv_apply_host(native_handle(), alpha, x, beta, y));
    }
    Info info() const {
        Info result{}; result.struct_size = sizeof(result);
        check(ispmv_query(native_handle(), &result)); return result;
    }
};
} // namespace ispmv
