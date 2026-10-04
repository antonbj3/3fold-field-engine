// ASan/UBSan control for src/field_engine/mesh_sdf_native_v1/field.cpp. Exact fields on two meshes
// (an axis box, and 140 stacked slabs giving 280 hits per column, past the 256-key column buffer)
// against a brute-force reference, identical outputs for 1 and 4 threads, untouched guard canaries,
// block classification and tile gather, and refusal before any output write for invalid arguments.
#undef NDEBUG   // the checks below call the kernel inside assert()
#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <vector>
#include "../src/field_engine/mesh_sdf_native_v1/field.cpp"

namespace {
const int F12[12][3] = {{0, 2, 1}, {0, 3, 2}, {4, 5, 6}, {4, 6, 7}, {0, 1, 5}, {0, 5, 4},
                        {2, 3, 7}, {2, 7, 6}, {1, 2, 6}, {1, 6, 5}, {3, 0, 4}, {3, 4, 7}};
const double PITCH = 0.5, JX = 0.5 * 1e-4 * 0.3819660112501051, JY = 0.5 * 1e-4 * 0.6180339887498949;

struct Box { double lo[3], hi[3]; };

void add_box(std::vector<double>& V, std::vector<int64_t>& T, const Box& b) {
    int64_t base = (int64_t)V.size() / 3;
    const double c[8][3] = {{b.lo[0], b.lo[1], b.lo[2]}, {b.hi[0], b.lo[1], b.lo[2]}, {b.hi[0], b.hi[1], b.lo[2]},
                            {b.lo[0], b.hi[1], b.lo[2]}, {b.lo[0], b.lo[1], b.hi[2]}, {b.hi[0], b.lo[1], b.hi[2]},
                            {b.hi[0], b.hi[1], b.hi[2]}, {b.lo[0], b.hi[1], b.hi[2]}};
    for (auto& p : c) V.insert(V.end(), p, p + 3);
    for (auto& f : F12) for (int k = 0; k < 3; ++k) T.push_back(base + f[k]);
}

template <class X> struct Guarded {   // one canary element on each side of the output
    std::vector<X> v; X canary;
    Guarded(int64_t n, X c) : v(n + 2, c), canary(c) {}
    X* data() { return v.data() + 1; }
    bool intact() const { return v.front() == canary && v.back() == canary; }
};

struct Case {
    std::vector<double> V; std::vector<int64_t> T; std::vector<Box> boxes; int64_t nx, ny, nz;
    int64_t N() const { return nx * ny * nz; }
};

struct Out {
    Guarded<uint8_t> solid, yta; Guarded<float> sd; Guarded<int64_t> stats; Guarded<double> ts;
    explicit Out(int64_t N) : solid(N, 0xA5), yta(N, 0xA5), sd(N, 971.f), stats(4, 971), ts(5, 971.) {}
    int run(const Case& c, int threads, const double* V = nullptr, const int64_t* T = nullptr,
            int64_t nv = -2, int64_t nt = -2, int64_t nx = -2, double pitch = PITCH, double ox = 0) {
        return meshsdf_field_cpu(V ? V : c.V.data(), nv == -2 ? (int64_t)c.V.size() / 3 : nv, T ? T : c.T.data(),
                                 nt == -2 ? (int64_t)c.T.size() / 3 : nt, ox, 0, 0, pitch, JX, JY,
                                 nx == -2 ? c.nx : nx, c.ny, c.nz, 1, threads, solid.data(), sd.data(), yta.data(),
                                 stats.data(), ts.data());
    }
    bool intact() const { return solid.intact() && yta.intact() && sd.intact() && stats.intact() && ts.intact(); }
};

// Brute-force reference: inside = sample point strictly inside a box (bounds sit a quarter pitch from
// every sample plane); surface = solid with a non-solid 6-neighbour or on the window border; distances
// by exhaustive search with the kernel's formula (pitch 0.5 makes every candidate exact).
void reference(const Case& c, std::vector<uint8_t>& s, std::vector<uint8_t>& y, std::vector<float>& sd) {
    const int64_t nx = c.nx, ny = c.ny, nz = c.nz, N = c.N();
    s.assign(N, 0); y.assign(N, 0); sd.assign(N, 0.f);
    for (int64_t i = 0; i < nx; ++i) for (int64_t j = 0; j < ny; ++j) for (int64_t k = 0; k < nz; ++k) {
        double p[3] = {i * PITCH + JX, j * PITCH + JY, k * PITCH};
        for (auto& b : c.boxes) {
            bool in = true;
            for (int a = 0; a < 3; ++a) in = in && b.lo[a] < p[a] && p[a] < b.hi[a];
            if (in) s[(i * ny + j) * nz + k] = 1;
        }
    }
    auto at = [&](int64_t i, int64_t j, int64_t k) { return s[(i * ny + j) * nz + k]; };
    for (int64_t i = 0; i < nx; ++i) for (int64_t j = 0; j < ny; ++j) for (int64_t k = 0; k < nz; ++k) {
        int64_t l = (i * ny + j) * nz + k;
        if (!s[l]) continue;
        bool border = i == 0 || j == 0 || k == 0 || i == nx - 1 || j == ny - 1 || k == nz - 1;
        y[l] = border || !at(i - 1, j, k) || !at(i + 1, j, k) || !at(i, j - 1, k) || !at(i, j + 1, k) ||
               !at(i, j, k - 1) || !at(i, j, k + 1);
    }
    for (int64_t l = 0; l < N; ++l) {
        int64_t i = l / (ny * nz), j = (l / nz) % ny, k = l % nz;
        double d[2] = {INFINITY, INFINITY};   // d[0]: to nearest solid, d[1]: to nearest non-solid
        for (int64_t m = 0; m < N; ++m) {
            double dx = (double)(m / (ny * nz) - i) * PITCH, dy = (double)((m / nz) % ny - j) * PITCH,
                   dz = (double)(m % nz - k) * PITCH;
            double e = std::sqrt((dx * dx + dy * dy) + dz * dz);
            int cls = s[m] ? 0 : 1;
            if (e < d[cls]) d[cls] = e;
        }
        float v = (float)d[0] - (float)d[1];
        float sg = (v > 0.f) ? 1.f : ((v < 0.f) ? -1.f : 0.f);
        sd[l] = v - sg * (float)(0.5 * PITCH);
    }
}
}  // namespace

int main() {
    std::vector<Case> cases(2);
    cases[0].boxes = {{{1.25, 1.25, 1.25}, {3.75, 3.25, 2.75}}};
    cases[0].nx = 10; cases[0].ny = 9; cases[0].nz = 8;
    for (int m = 0; m < 140; ++m) cases[1].boxes.push_back({{0.25, 0.25, m + 1.3}, {1.25, 1.25, m + 1.7}});
    cases[1].nx = 4; cases[1].ny = 4; cases[1].nz = 286;
    int exact = 0, controls = 0;
    for (auto& c : cases) {
        for (auto& b : c.boxes) add_box(c.V, c.T, b);
        std::vector<uint8_t> rs, ry; std::vector<float> rsd;
        reference(c, rs, ry, rsd);
        Out one(c.N()), four(c.N());
        assert(one.run(c, 1) == 0 && four.run(c, 4) == 0);
        assert(one.intact() && four.intact());
        assert(std::memcmp(one.solid.data(), rs.data(), c.N()) == 0);
        assert(std::memcmp(one.yta.data(), ry.data(), c.N()) == 0);
        assert(std::memcmp(one.sd.data(), rsd.data(), c.N() * sizeof(float)) == 0);
        assert(std::memcmp(one.solid.data(), four.solid.data(), c.N()) == 0);
        assert(std::memcmp(one.yta.data(), four.yta.data(), c.N()) == 0);
        assert(std::memcmp(one.sd.data(), four.sd.data(), c.N() * sizeof(float)) == 0);
        assert(std::memcmp(one.stats.data(), four.stats.data(), 4 * sizeof(int64_t)) == 0);
        assert(one.stats.data()[2] == 0);   // winding and parity agree on disjoint closed boxes
        ++exact;
        // block classification and tile gather on the computed field, 1 and 4 threads
        const int64_t block = 4, nb = ((c.nx + 3) / 4) * ((c.ny + 3) / 4) * ((c.nz + 3) / 4);
        Guarded<int32_t> k1(nb, 971), k4(nb, 971);
        assert(meshsdf_classify_cpu(one.sd.data(), c.nx, c.ny, c.nz, block, 3.4641016f, k1.data(), 1) == 0);
        assert(meshsdf_classify_cpu(one.sd.data(), c.nx, c.ny, c.nz, block, 3.4641016f, k4.data(), 4) == 0);
        assert(k1.intact() && k4.intact() && std::memcmp(k1.data(), k4.data(), nb * sizeof(int32_t)) == 0);
        std::vector<int32_t> active;
        for (int64_t b = 0; b < nb; ++b) if (k1.data()[b] == 0) active.push_back((int32_t)b);
        assert(!active.empty());
        Guarded<float> tiles((int64_t)active.size() * 64, 971.f);
        assert(meshsdf_tiles_cpu(one.sd.data(), c.nx, c.ny, c.nz, block, active.data(), (int64_t)active.size(),
                                 tiles.data(), 4) == 0 && tiles.intact());
        const int64_t nby = (c.ny + 3) / 4, nbz = (c.nz + 3) / 4;
        for (size_t a = 0; a < active.size(); ++a) {
            int64_t bi = active[a] / (nby * nbz), bj = (active[a] / nbz) % nby, bk = active[a] % nbz;
            for (int64_t t = 0; t < 64; ++t) {
                int64_t vx = std::min(bi * 4 + t / 16, c.nx - 1), vy = std::min(bj * 4 + (t / 4) % 4, c.ny - 1),
                        vz = std::min(bk * 4 + t % 4, c.nz - 1);
                assert(tiles.data()[a * 64 + t] == one.sd.data()[(vx * c.ny + vy) * c.nz + vz]);
            }
        }
    }
    // Invalid arguments: nonzero status and every output byte unchanged.
    const Case& c = cases[0];
    Out bad(c.N());
    auto untouched = [&]() {
        for (int64_t l = 0; l < c.N(); ++l)
            if (bad.solid.data()[l] != 0xA5 || bad.yta.data()[l] != 0xA5 || bad.sd.data()[l] != 971.f) return false;
        return bad.intact() && bad.stats.data()[0] == 971 && bad.ts.data()[0] == 971.;
    };
    std::vector<int64_t> T = c.T;
    T[7] = (int64_t)c.V.size() / 3;
    assert(bad.run(c, 4, nullptr, T.data()) == 2 && untouched()); ++controls;
    T[7] = -1;
    assert(bad.run(c, 4, nullptr, T.data()) == 2 && untouched()); ++controls;
    T[7] = std::numeric_limits<int64_t>::min();
    assert(bad.run(c, 4, nullptr, T.data()) == 2 && untouched()); ++controls;
    assert(bad.run(c, 4, nullptr, nullptr, -2, 0) == 1 && untouched()); ++controls;
    assert(bad.run(c, 4, nullptr, nullptr, 0) == 1 && untouched()); ++controls;
    assert(bad.run(c, 4, nullptr, nullptr, -2, -2, 0) == 1 && untouched()); ++controls;
    assert(bad.run(c, 4, nullptr, nullptr, -2, -2, std::numeric_limits<int64_t>::max()) == 1 && untouched()); ++controls;
    assert(bad.run(c, 4, nullptr, nullptr, -2, -2, -2, 0.0) == 1 && untouched()); ++controls;
    assert(bad.run(c, 4, nullptr, nullptr, -2, -2, -2, std::nan("")) == 1 && untouched()); ++controls;
    assert(bad.run(c, 4, nullptr, nullptr, -2, -2, -2, PITCH, INFINITY) == 1 && untouched()); ++controls;
    assert(meshsdf_field_cpu(nullptr, 8, c.T.data(), 12, 0, 0, 0, PITCH, JX, JY, c.nx, c.ny, c.nz, 1, 4,
                             bad.solid.data(), bad.sd.data(), bad.yta.data(), bad.stats.data(), bad.ts.data()) == 1);
    assert(untouched()); ++controls;
    assert(meshsdf_field_cpu(c.V.data(), 8, c.T.data(), 12, 0, 0, 0, PITCH, JX, JY, c.nx, c.ny, c.nz, 1, 4,
                             bad.solid.data(), nullptr, bad.yta.data(), bad.stats.data(), bad.ts.data()) == 1);
    assert(untouched()); ++controls;
    std::vector<float> dense(c.N(), 1.f);
    Guarded<int32_t> kind(8, 971);
    assert(meshsdf_classify_cpu(dense.data(), c.nx, c.ny, c.nz, 0, 1.f, kind.data(), 4) == 1); ++controls;
    assert(meshsdf_classify_cpu(nullptr, c.nx, c.ny, c.nz, 4, 1.f, kind.data(), 4) == 1); ++controls;
    assert(meshsdf_classify_cpu(dense.data(), 0, c.ny, c.nz, 4, 1.f, kind.data(), 4) == 1); ++controls;
    assert(kind.intact() && kind.data()[0] == 971);
    Guarded<float> tiles(64, 971.f);
    int32_t ids[2] = {0, 18};   // 3*3*2 = 18 blocks at block 4: id 18 is one past the end
    assert(meshsdf_tiles_cpu(dense.data(), c.nx, c.ny, c.nz, 4, ids, 2, tiles.data(), 4) == 2); ++controls;
    ids[1] = -1;
    assert(meshsdf_tiles_cpu(dense.data(), c.nx, c.ny, c.nz, 4, ids, 2, tiles.data(), 4) == 2); ++controls;
    assert(meshsdf_tiles_cpu(dense.data(), c.nx, c.ny, c.nz, 4, ids, -1, tiles.data(), 4) == 1); ++controls;
    assert(meshsdf_tiles_cpu(dense.data(), c.nx, c.ny, c.nz, 4, nullptr, 1, tiles.data(), 4) == 1); ++controls;
    for (int t = 0; t < 64; ++t) assert(tiles.data()[t] == 971.f);
    assert(tiles.intact());
    std::cout << "{\"cases\":" << exact << ",\"controls\":" << controls
              << ",\"exact\":true,\"sanitizers\":\"address,undefined\"}\n";
    return 0;
}
