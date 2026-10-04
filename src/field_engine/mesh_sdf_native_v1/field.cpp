// Native CPU backend for the mesh->SDF field stage: the same binary64 operation sequence as
// faltkarna_v1_mesh_to_sdf (raypar_vindning + two SciPy EDTs + half-pitch correction + surface) and
// the same block classifier, threaded with OpenMP. Built explicitly (see mesh_sdf_native_v1.py):
// -O3 -fopenmp -fno-fast-math -ffp-contract=off. No fast-math and no contraction: every comparison
// and rounding must equal NumPy's. Every entry point validates its arguments and returns nonzero
// before writing any output when they are outside the contract.
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>
#include <algorithm>
#include <limits>
#include <omp.h>

static const int BIG = 1 << 30;

struct Hit { int64_t col; double z; int8_t s; };

extern "C" int meshsdf_field_cpu(const double* V, int64_t nv, const int64_t* T, int64_t nt,
                                 double ox, double oy, double oz, double pitch, double jx, double jy,
                                 int64_t nx, int64_t ny, int64_t nz, int ytkorr, int nthreads,
                                 uint8_t* solid, float* sd, uint8_t* yta, int64_t* stats, double* tstage) {
    // 1: pointer/count/grid/scalar contract; 2: face index outside [0, nv). Nothing is written then.
    const int64_t CELLS_MAX = (int64_t)std::numeric_limits<int32_t>::max() - 1;
    if (!V || !T || !solid || !sd || !yta || !stats || !tstage) return 1;
    if (nv < 1 || nt < 1 || nt > CELLS_MAX || nv > std::numeric_limits<int64_t>::max() / 3) return 1;
    if (nx < 1 || ny < 1 || nz < 1 || nx > CELLS_MAX / ny || nx * ny > CELLS_MAX / nz) return 1;
    if (!(std::isfinite(pitch) && pitch > 0) || !std::isfinite(ox) || !std::isfinite(oy) || !std::isfinite(oz) ||
        !std::isfinite(jx) || !std::isfinite(jy)) return 1;
    for (int64_t t = 0; t < 3 * nt; ++t) if (T[t] < 0 || T[t] >= nv) return 2;
    if (nthreads > 0) omp_set_num_threads(nthreads);
    const int64_t ncol = nx * ny, N = ncol * nz;
    double t0 = omp_get_wtime();
    // ---- 1. triangle -> column hits (per-thread buffers) ----
    int nth = omp_get_max_threads();
    std::vector<std::vector<Hit>> buf(nth);
    int64_t npairs = 0;
#pragma omp parallel reduction(+:npairs)
    {
        std::vector<Hit>& out = buf[omp_get_thread_num()];
#pragma omp for schedule(dynamic, 16)
        for (int64_t t = 0; t < nt; ++t) {
            const double* A = V + 3 * T[3 * t];
            const double* B = V + 3 * T[3 * t + 1];
            const double* C = V + 3 * T[3 * t + 2];
            double e1x = B[0] - A[0], e1y = B[1] - A[1], e2x = C[0] - A[0], e2y = C[1] - A[1];
            double a2 = e1x * e2y - e1y * e2x;
            double skala = std::max(std::max(std::fabs(e1x), std::fabs(e1y)), std::max(std::fabs(e2x), std::fabs(e2y)));
            if (!(std::fabs(a2) > 1e-12 * skala * skala)) continue;
            double xmin = std::min(std::min(A[0], B[0]), C[0]), xmax = std::max(std::max(A[0], B[0]), C[0]);
            double ymin = std::min(std::min(A[1], B[1]), C[1]), ymax = std::max(std::max(A[1], B[1]), C[1]);
            int64_t i0 = (int64_t)std::ceil(((xmin - jx) - ox) / pitch - 0.0);
            int64_t i1 = (int64_t)std::floor(((xmax - jx) - ox) / pitch - 0.0);
            int64_t j0 = (int64_t)std::ceil(((ymin - jy) - oy) / pitch - 0.0);
            int64_t j1 = (int64_t)std::floor(((ymax - jy) - oy) / pitch - 0.0);
            i0 = std::min(std::max(i0, (int64_t)0), nx); i1 = std::min(std::max(i1, (int64_t)-1), nx - 1);
            j0 = std::min(std::max(j0, (int64_t)0), ny); j1 = std::min(std::max(j1, (int64_t)-1), ny - 1);
            if (i1 < i0 || j1 < j0) continue;
            npairs += (i1 - i0 + 1) * (j1 - j0 + 1);
            double sg = (a2 > 0) ? 1.0 : ((a2 < 0) ? -1.0 : 0.0);
            int8_t tk = (int8_t)((a2 > 0) ? 1 : ((a2 < 0) ? -1 : 0));
            double ax = A[0], ay = A[1], az = A[2], bx = B[0], by = B[1], bz = B[2], cx = C[0], cy = C[1], cz = C[2];
            double e3 = std::max(std::max(std::fabs(bx - ax), std::fabs(by - ay)), std::fabs(bz - az));
            double e4 = std::max(std::max(std::fabs(cx - ax), std::fabs(cy - ay)), std::fabs(cz - az));
            bool local = std::fabs(az) > 1024.0 * std::max(e3, e4);
            for (int64_t ii = i0; ii <= i1; ++ii) {
                double px = (ox + ((double)ii + 0.0) * pitch) + jx;
                for (int64_t jj = j0; jj <= j1; ++jj) {
                    double py = (oy + ((double)jj + 0.0) * pitch) + jy;
                    double l0 = (bx - px) * (cy - py) - (by - py) * (cx - px);
                    if (!(l0 * sg >= 0)) continue;
                    double l1 = (cx - px) * (ay - py) - (cy - py) * (ax - px);
                    if (!(l1 * sg >= 0)) continue;
                    double l2 = (ax - px) * (by - py) - (ay - py) * (bx - px);
                    if (!(l2 * sg >= 0)) continue;
                    double z;
                    if (local) z = az + (l1 * (bz - az) + l2 * (cz - az)) / a2;
                    else z = ((l0 * az + l1 * bz) + l2 * cz) / a2;
                    out.push_back(Hit{ii * ny + jj, z, tk});
                }
            }
        }
    }
    double t1 = omp_get_wtime();
    // ---- 2. CSR by column, global z range ----
    std::vector<int32_t> cnt(ncol + 1, 0);
    double zmin = INFINITY, zmax = -INFINITY;
    int64_t nh = 0;
    for (auto& b : buf) nh += (int64_t)b.size();
    if (nh > CELLS_MAX) return 3;   // hit offsets are int32; outputs not yet written
    for (auto& b : buf) for (auto& h : b) { cnt[h.col + 1]++; zmin = std::min(zmin, h.z); zmax = std::max(zmax, h.z); }
    for (int64_t c = 0; c < ncol; ++c) cnt[c + 1] += cnt[c];
    std::vector<int32_t> fill(cnt.begin(), cnt.end() - 1);
    std::vector<double> hz(std::max<int64_t>(nh, 1));
    std::vector<int8_t> hs(std::max<int64_t>(nh, 1));
    for (auto& b : buf) for (auto& h : b) { int32_t k = fill[h.col]++; hz[k] = h.z; hs[k] = h.s; }
    double span = std::max(zmax - zmin, 1e-12);
    int64_t ndiff = 0;
    double t2 = omp_get_wtime();
    // ---- 3. per-voxel winding ----
    if (nh == 0) { std::memset(solid, 0, N); }
    else {
#pragma omp parallel for schedule(static) reduction(+:ndiff)
        for (int64_t c = 0; c < ncol; ++c) {
            uint8_t* sc = solid + c * nz;
            int32_t h0 = cnt[c], h1 = cnt[c + 1];
            if (h0 == h1) { std::memset(sc, 0, nz); continue; }
            double fc = (double)c;
            double keys[256]; int kn = std::min(h1 - h0, 256);
            for (int h = 0; h < kn; ++h) keys[h] = (fc + 0.25) + (0.5 * (hz[h0 + h] - zmin)) / span;
            for (int64_t k = 0; k < nz; ++k) {
                double zc = oz + ((double)k + 0.0) * pitch;
                double v = 0.25 + (0.5 * (zc - zmin)) / span;
                double kc = fc + std::min(std::max(v, 0.0), 0.999);
                int w = 0, np_ = 0;
                for (int32_t h = h0; h < h1; ++h) {
                    double key = (h - h0 < 256) ? keys[h - h0] : (fc + 0.25) + (0.5 * (hz[h] - zmin)) / span;
                    if (key > kc) { w += hs[h]; np_++; }
                }
                uint8_t s = (w != 0);
                sc[k] = s;
                ndiff += ((np_ & 1) != s);
            }
        }
    }
    double t3 = omp_get_wtime();
    // ---- 4. exact EDT, both classes, SciPy distance formula ----
    std::vector<int32_t> dz(N), cyv(N), czv(N);
    std::vector<float> dist[2] = {std::vector<float>(N), std::vector<float>(N)};
    for (int cls = 0; cls < 2; ++cls) {
        const uint8_t feat = (cls == 0) ? 1 : 0;   // d_out: features = solid; d_in: features = non-solid
#pragma omp parallel for schedule(static)
        for (int64_t l = 0; l < ncol; ++l) {
            const uint8_t* m = solid + l * nz; int32_t* o = dz.data() + l * nz;
            int64_t last = -1;
            for (int64_t k = 0; k < nz; ++k) { if (m[k] == feat) last = k; o[k] = (last >= 0) ? (int32_t)(last - k) : BIG; }
            int64_t nxt = -1;
            for (int64_t k = nz - 1; k >= 0; --k) {
                if (m[k] == feat) nxt = k;
                if (nxt >= 0) { int32_t dn = (int32_t)(nxt - k); if (o[k] == BIG || dn < -o[k]) o[k] = dn; }
            }
        }
        // axis 1: lines (i, k); gather to contiguous buffers
#pragma omp parallel
        {
            std::vector<int32_t> g(ny), v(ny + 1); std::vector<double> zb(ny + 2); std::vector<double> gv(ny);
#pragma omp for schedule(static)
            for (int64_t l = 0; l < nx * nz; ++l) {
                int64_t i = l / nz, k = l % nz;
                for (int64_t j = 0; j < ny; ++j) g[j] = dz[(i * ny + j) * nz + k];
                int kk = -1;
                for (int64_t j = 0; j < ny; ++j) {
                    if (g[j] == BIG) continue;
                    double gj = (double)g[j] * (double)g[j]; gv[j] = gj;
                    if (kk < 0) { kk = 0; v[0] = (int32_t)j; zb[0] = -INFINITY; zb[1] = INFINITY; continue; }
                    while (true) {
                        int32_t q = v[kk]; double fq = (double)j, fv = (double)q;
                        double s = ((gj + fq * fq) - (gv[q] + fv * fv)) / (2.0 * fq - 2.0 * fv);
                        if (s <= zb[kk]) { kk--; if (kk < 0) { kk = 0; v[0] = (int32_t)j; zb[0] = -INFINITY; zb[1] = INFINITY; break; } }
                        else { kk++; v[kk] = (int32_t)j; zb[kk] = s; zb[kk + 1] = INFINITY; break; }
                    }
                }
                if (kk < 0) { for (int64_t j = 0; j < ny; ++j) { cyv[(i * ny + j) * nz + k] = BIG; czv[(i * ny + j) * nz + k] = BIG; } continue; }
                int m_ = 0;
                for (int64_t j = 0; j < ny; ++j) {
                    while (zb[m_ + 1] < (double)j) m_++;
                    int32_t q = v[m_];
                    cyv[(i * ny + j) * nz + k] = (int32_t)(q - j); czv[(i * ny + j) * nz + k] = g[q];
                }
            }
        }
        // axis 0: lines (j, k)
        float* D = dist[cls].data();
#pragma omp parallel
        {
            std::vector<int32_t> a(nx), b(nx), v(nx + 1); std::vector<double> zb(nx + 2), gv(nx);
#pragma omp for schedule(static)
            for (int64_t l = 0; l < ny * nz; ++l) {
                int64_t st = ny * nz;
                for (int64_t i = 0; i < nx; ++i) { a[i] = cyv[i * st + l]; b[i] = czv[i * st + l]; }
                int kk = -1;
                for (int64_t i = 0; i < nx; ++i) {
                    if (a[i] == BIG) continue;
                    double gi = (double)a[i] * (double)a[i] + (double)b[i] * (double)b[i]; gv[i] = gi;
                    if (kk < 0) { kk = 0; v[0] = (int32_t)i; zb[0] = -INFINITY; zb[1] = INFINITY; continue; }
                    while (true) {
                        int32_t q = v[kk]; double fq = (double)i, fv = (double)q;
                        double s = ((gi + fq * fq) - (gv[q] + fv * fv)) / (2.0 * fq - 2.0 * fv);
                        if (s <= zb[kk]) { kk--; if (kk < 0) { kk = 0; v[0] = (int32_t)i; zb[0] = -INFINITY; zb[1] = INFINITY; break; } }
                        else { kk++; v[kk] = (int32_t)i; zb[kk] = s; zb[kk + 1] = INFINITY; break; }
                    }
                }
                if (kk < 0) { for (int64_t i = 0; i < nx; ++i) D[i * st + l] = -1.0f; continue; }
                int m_ = 0;
                for (int64_t i = 0; i < nx; ++i) {
                    while (zb[m_ + 1] < (double)i) m_++;
                    int32_t q = v[m_];
                    double dx = (double)(q - i) * pitch, dy = (double)a[q] * pitch, dzz = (double)b[q] * pitch;
                    D[i * st + l] = (float)std::sqrt((dx * dx + dy * dy) + dzz * dzz);
                }
            }
        }
    }
    double t4 = omp_get_wtime();
    // ---- 5. sd and surface ----
    const float half = (float)(0.5 * pitch);
    int64_t nsolid = 0;
#pragma omp parallel for schedule(static) reduction(+:nsolid)
    for (int64_t lin = 0; lin < N; ++lin) {
        float v = dist[0][lin] - dist[1][lin];
        if (ytkorr) { float sg = (v > 0.f) ? 1.f : ((v < 0.f) ? -1.f : 0.f); v = v - sg * half; }
        sd[lin] = v;
        uint8_t s = solid[lin]; nsolid += s;
        uint8_t y = 0;
        if (s) {
            int64_t k = lin % nz, j = (lin / nz) % ny, i = lin / (nz * ny);
            if (i == 0 || i == nx - 1 || j == 0 || j == ny - 1 || k == 0 || k == nz - 1) y = 1;
            else if (!solid[lin - ny * nz] || !solid[lin + ny * nz] || !solid[lin - nz] || !solid[lin + nz] || !solid[lin - 1] || !solid[lin + 1]) y = 1;
        }
        yta[lin] = y;
    }
    double t5 = omp_get_wtime();
    stats[0] = npairs; stats[1] = nh; stats[2] = ndiff; stats[3] = nsolid;
    tstage[0] = t1 - t0; tstage[1] = t2 - t1; tstage[2] = t3 - t2; tstage[3] = t4 - t3; tstage[4] = t5 - t4;
    return 0;
}

// Block classifier: the reference's k_classify_dense / k_eval_tile_dense (binary32 compares), threaded.
extern "C" int meshsdf_classify_cpu(const float* dense, int64_t nx, int64_t ny, int64_t nz, int64_t block,
                                    float margin, int32_t* kind, int nthreads) {
    if (!dense || !kind || nx < 1 || ny < 1 || nz < 1 || block < 1 || block > 1024 ||
        nx > std::numeric_limits<int64_t>::max() / ny / nz) return 1;
    if (nthreads > 0) omp_set_num_threads(nthreads);
    int64_t nbx = (nx + block - 1) / block, nby = (ny + block - 1) / block, nbz = (nz + block - 1) / block;
    int64_t nb = nbx * nby * nbz;
#pragma omp parallel for schedule(static)
    for (int64_t tid = 0; tid < nb; ++tid) {
        int64_t bk = tid % nbz, bj = (tid / nbz) % nby, bi = tid / (nbz * nby);
        int64_t vx0 = bi * block, vy0 = bj * block, vz0 = bk * block;
        float min_abs = 1.0e30f; bool pos = false, neg = false;
        for (int c = 0; c < 8; ++c) {
            int64_t cx = std::min(vx0 + (c & 1) * block, nx - 1);
            int64_t cy = std::min(vy0 + ((c >> 1) & 1) * block, ny - 1);
            int64_t cz = std::min(vz0 + ((c >> 2) & 1) * block, nz - 1);
            float v = dense[(cx * ny + cy) * nz + cz];
            min_abs = std::min(min_abs, std::fabs(v));
            if (v >= 0.0f) pos = true; else neg = true;
        }
        kind[tid] = (pos && neg) ? 0 : ((min_abs <= margin) ? 0 : (neg ? -1 : 1));
    }
    return 0;
}

extern "C" int meshsdf_tiles_cpu(const float* dense, int64_t nx, int64_t ny, int64_t nz, int64_t block,
                                 const int32_t* active, int64_t nact, float* out, int nthreads) {
    if (!dense || nx < 1 || ny < 1 || nz < 1 || block < 1 || block > 1024 || nact < 0 ||
        nx > std::numeric_limits<int64_t>::max() / ny / nz || (nact > 0 && (!active || !out))) return 1;
    int64_t nby = (ny + block - 1) / block, nbz = (nz + block - 1) / block, b3 = block * block * block;
    const int64_t nb = ((nx + block - 1) / block) * nby * nbz;
    for (int64_t a = 0; a < nact; ++a) if (active[a] < 0 || active[a] >= nb) return 2;
    if (nthreads > 0) omp_set_num_threads(nthreads);
#pragma omp parallel for schedule(static)
    for (int64_t a = 0; a < nact; ++a) {
        int64_t lin = active[a];
        int64_t bk = lin % nbz, bi = lin / (nbz * nby), bj = (lin / nbz) % nby;
        float* o = out + a * b3;
        for (int64_t li = 0; li < block; ++li) for (int64_t lj = 0; lj < block; ++lj) for (int64_t lk = 0; lk < block; ++lk) {
            int64_t vx = std::min(bi * block + li, nx - 1), vy = std::min(bj * block + lj, ny - 1), vz = std::min(bk * block + lk, nz - 1);
            *o++ = dense[(vx * ny + vy) * nz + vz];
        }
    }
    return 0;
}
