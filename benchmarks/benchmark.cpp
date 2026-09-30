#include "measurement.hpp"

// Public C API client: no private plan descriptors or implementation headers.
int main(int argc, char** argv) {
    if (argc != 5) {
        std::cerr << "usage: ispmv_benchmark LABEL A_CSR AT_CSR METHOD\n"
                     "methods: ispmv cusparse_alg2 spmvop_alg1 spmvop_alg2\n";
        return 2;
    }
    try {
        const std::string method = argv[4];
        const bool native = method == "ispmv";
        if (!native && method != "cusparse_alg2" && method != "spmvop_alg1" && method != "spmvop_alg2")
            throw std::invalid_argument("unknown method");
        HostCsr a, at;
        if (!load_csr(argv[2], a) || !load_csr(argv[3], at)) return 2;
        if (a.rows != at.cols || a.cols != at.rows || a.nnz != at.nnz)
            throw std::invalid_argument("A/AT dimensions or nonzero counts differ");
        Operator left, right;
        left.method = right.method = native ? "ispmv" : method;
        left.upload(a); right.upload(at,left.hardware);
        Events events;
        CUDA_CHECK(cudaDeviceSynchronize());
        const auto start = std::chrono::steady_clock::now();
        double a_ms = 0, at_ms = 0;
        a_ms = left.prepare(a); at_ms = right.prepare(at);
        CUDA_CHECK(cudaDeviceSynchronize());
        const double total_ms = elapsed_ms(start);
        const char* preparation = "separate";
        auto ar = measure_operator(left, a, events, a_ms, argv[1], argv[2], preparation);
        auto tr = measure_operator(right, at, events, at_ms, argv[1], argv[3], preparation);
        ar["direction"] = "A"; tr["direction"] = "AT";
        ar["method"] = method; tr["method"] = method;
        Json result = Json::Object{{"schema",3},{"label",argv[1]},{"method",method},
            {"preparation",preparation},{"plan_build_wall_ms",total_ms},
            {"separate_a_ms",Json(a_ms)},{"separate_at_ms",Json(at_ms)},
            {"device_initialization_ms",left.device_initialization_ms+right.device_initialization_ms},{"device_csr_ready",true},{"copy_device_matrix",false},
            {"ispmv_abi",int(ispmv_version())},{"directions",Json::Array{ar,tr}}};
        std::cout << "COMPARISON_RESULT "; result.write(std::cout);
        std::cout << '\n' << std::flush;
        return 0; // Numerical acceptance is explicit in each JSON mode.
    } catch (const std::exception& error) {
        std::cerr << "BENCHMARK_ERROR " << error.what() << '\n';
        return 4;
    }
}
