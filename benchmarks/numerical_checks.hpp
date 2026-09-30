#pragma once
#include "json.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

// Fixed acceptance gates. Diagnostics separate the mixed
// forward requirement from scaled backward error; neither maximum identifies
// the other's worst row. Record row-local evidence for error analysis.
static Json check_result(const std::vector<double>&y,const std::vector<double>&again,
                         const std::vector<long double>&ref,const std::vector<long double>&l1,
                         int trial) {
    if (again.size()!=y.size() || ref.size()!=y.size() || l1.size()!=y.size())
        throw std::invalid_argument("numerical check array lengths disagree");
    double relative=0,backward=0,absolute=0;
    bool finite=true,reference_finite=true;
    const size_t absent=std::numeric_limits<size_t>::max();
    size_t relative_row=absent,backward_row=absent;
    for(size_t r=0;r<y.size();++r) {
        const bool output_ok=std::isfinite(y[r]) && std::isfinite(again[r]);
        const bool reference_ok=std::isfinite(ref[r]) && std::isfinite(l1[r]) && l1[r]>=0;
        finite &= output_ok;reference_finite &= reference_ok;
        if(!output_ok || !reference_ok) continue;
        const long double err=std::abs((long double)y[r]-ref[r]);
        absolute=std::max(absolute,double(err));
        const double forward=double(err/std::max(1.L,std::abs(ref[r])));
        const double scaled=double(err/std::max(1.L,l1[r]));
        if(relative_row==absent || forward>relative) {relative=forward;relative_row=r;}
        if(backward_row==absent || scaled>backward) {backward=scaled;backward_row=r;}
    }
    const bool repeatable=y.empty() || std::memcmp(y.data(),again.data(),y.size()*8)==0;
    const bool backward_pass=reference_finite && backward<=1e-13;
    const bool forward_pass=reference_finite && relative<=1e-11;
    const bool accepted=finite && repeatable && backward_pass && (trial==6 || forward_pass);
    auto number=[](double x){return std::isfinite(x)?Json(x):Json();};
    auto row_evidence=[&](size_t row)->Json {
        if(row==absent) return Json();
        const long double den=std::max(1.L,std::abs(ref[row]));
        return Json::Object{{"row",int64_t(row)},{"computed",number(y[row])},
            {"reference",number(double(ref[row]))},{"l1_scale",number(double(l1[row]))},
            {"absolute",number(double(std::abs((long double)y[row]-ref[row])))},
            {"mixed_condition",number(double(std::max(1.L,l1[row])/den))}};
    };
    return Json::Object{{"trial",trial},{"finite",finite},{"repeatable",repeatable},
        {"relative",number(relative)},{"backward",number(backward)},{"absolute",number(absolute)},
        {"accepted",accepted},{"relative_gate_applies",trial!=6},
        {"reference_finite",reference_finite},{"reference_mantissa_bits",std::numeric_limits<long double>::digits},
        {"backward_gate_passed",backward_pass},{"forward_tolerance_met",forward_pass},
        {"forward_metric","max |error|/max(1,|reference|); not pure relative error"},
        {"worst_forward_row",row_evidence(relative_row)},
        {"worst_backward_row",row_evidence(backward_row)}};
}
