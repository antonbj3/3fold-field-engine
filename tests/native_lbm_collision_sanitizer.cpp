#include <array>
#include <cassert>
#include <cstdint>
#include <limits>
extern "C" int field_lbm_collision(std::int64_t, double, double,
    const double*, const double*, const double*, const double*,
    const double*, const double*, const double*, double*);
int main() {
    std::array<double, 38> f, cu{}, ea{};
    std::array<double, 2> rho{2.0, 2.0}, zero{};
    std::array<double, 19> weights;
    std::array<double, 40> out;
    f.fill(3.0); weights.fill(1.0); out.fill(123.0);
    auto call = [&](std::int64_t n, double tau, double pref, const double* input) {
        return field_lbm_collision(n, tau, pref, input, rho.data(), cu.data(),
            zero.data(), ea.data(), zero.data(), weights.data(), out.data() + 1);
    };
    assert(call(2, 2.0, 0.0, f.data()) == 0);
    for (int i = 1; i <= 38; ++i) assert(out[i] == 2.5);
    assert(out.front() == 123.0 && out.back() == 123.0);
    const auto saved = out;
    assert(call(-1, 2.0, 0.0, f.data()) == 1);
    assert(call(std::numeric_limits<std::int64_t>::max(), 2.0, 0.0, f.data()) == 1);
    assert(call(2, 0.0, 0.0, f.data()) == 1);
    assert(call(2, std::numeric_limits<double>::quiet_NaN(), 0.0, f.data()) == 1);
    assert(call(2, 2.0, std::numeric_limits<double>::infinity(), f.data()) == 1);
    assert(call(2, 2.0, 0.0, nullptr) == 2);
    assert(call(0, 2.0, 0.0, nullptr) == 0);
    assert(out == saved);
}
