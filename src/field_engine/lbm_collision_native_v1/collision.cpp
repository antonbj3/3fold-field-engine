#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

// Compile without fast-math or contraction: every intermediate matches the
// original NumPy expression order, including division by tau.
extern "C" int field_lbm_collision(
    std::int64_t cells, double tau, double pref,
    const double* f, const double* rho, const double* cu,
    const double* usq, const double* ea, const double* ua,
    const double* weights, double* out) {
    if (cells < 0 || static_cast<std::uint64_t>(cells) >
            std::numeric_limits<std::size_t>::max() / (19 * sizeof(double)) ||
            !std::isfinite(tau) || tau == 0.0 || !std::isfinite(pref)) return 1;
    if (cells == 0) return 0;
    if (!f || !rho || !cu || !usq || !ea || !ua || !weights || !out) return 2;
    double pw[19];
    for (int q = 0; q < 19; ++q) pw[q] = pref * weights[q];
    for (std::int64_t cell = 0; cell < cells; ++cell) {
        for (int q = 0; q < 19; ++q) {
            const auto i = static_cast<std::size_t>(cell) * 19 + q;
            const double c = cu[i];
            const double equilibrium = (weights[q] * rho[cell]) *
                (((1.0 + 3.0 * c) + 4.5 * (c * c)) - 1.5 * usq[cell]);
            const double forcing = (pw[q] * rho[cell]) *
                (3.0 * (ea[i] - ua[cell]) + (9.0 * c) * ea[i]);
            out[i] = (f[i] - ((f[i] - equilibrium) / tau)) + forcing;
        }
    }
    return 0;
}

#include "moments.hpp"
