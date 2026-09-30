#pragma once
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <map>
#include <ostream>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

// Output-only JSON writer for benchmark results.
struct Json {
    using Array = std::vector<Json>;
    using Object = std::map<std::string, Json>;
    std::variant<std::nullptr_t, bool, int64_t, double, std::string, Array, Object> value;
    Json() : value(nullptr) {}
    Json(bool v) : value(v) {}
    Json(int v) : value(int64_t(v)) {}
    Json(int64_t v) : value(v) {}
    Json(size_t v) : value(int64_t(v)) {}
    Json(double v) : value(v) {
        if (!std::isfinite(v))
            throw std::runtime_error("nonfinite JSON number");
    }
    Json(const char *v) : value(std::string(v)) {}
    Json(std::string v) : value(std::move(v)) {}
    Json(Array v) : value(std::move(v)) {}
    Json(Object v) : value(std::move(v)) {}
    Json &operator[](const std::string &k) {
        if (!std::holds_alternative<Object>(value))
            value = Object{};
        return std::get<Object>(value)[k];
    }
    static void string(std::ostream &s, const std::string &v) {
        s << '"';
        for (unsigned char c : v) {
            if (c == '"' || c == '\\')
                s << '\\' << c;
            else if (c < 32)
                s << "\\u00" << "0123456789abcdef"[c >> 4] << "0123456789abcdef"[c & 15];
            else
                s << c;
        }
        s << '"';
    }
    void write(std::ostream &s) const {
        std::visit(
            [&](const auto &v) {
                using T = std::decay_t<decltype(v)>;
                if constexpr (std::is_same_v<T, std::nullptr_t>)
                    s << "null";
                else if constexpr (std::is_same_v<T, bool>)
                    s << (v ? "true" : "false");
                else if constexpr (std::is_same_v<T, std::string>)
                    string(s, v);
                else if constexpr (std::is_same_v<T, Array>) {
                    s << '[';
                    bool first = true;
                    for (const auto &x : v) {
                        if (!first)
                            s << ',';
                        first = false;
                        x.write(s);
                    }
                    s << ']';
                } else if constexpr (std::is_same_v<T, Object>) {
                    s << '{';
                    bool first = true;
                    for (const auto &x : v) {
                        if (!first)
                            s << ',';
                        first = false;
                        string(s, x.first);
                        s << ':';
                        x.second.write(s);
                    }
                    s << '}';
                } else
                    s << std::setprecision(17) << v;
            },
            value);
    }
};
