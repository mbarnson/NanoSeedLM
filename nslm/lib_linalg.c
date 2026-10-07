// nslm/lib_linalg.c - the dense linear algebra of GPTQ-style error feedback (S-01 S3), in double, C99 + Clang vectors.
//
// GPTQ needs U, the upper Cholesky factor of the inverse Hessian: H^-1 = U^T U.  With J the order reversal and
// J H J = M M^T (lower Cholesky), H = (J M J)(J M J)^T, so H^-1 = (J M^-1 J)^T (J M^-1 J): U = J M^-1 J, upper
// triangular with a positive diagonal.  One Cholesky and one triangular inverse, 2n^3/3 flops.
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "linalg.h"

typedef double d4 __attribute__((ext_vector_type(4)));
static inline d4 ld4(const double* p) { d4 v; memcpy(&v, p, sizeof v); return v; }   // unaligned

// In-place lower Cholesky of a (n x n, row-major, symmetric): a = L L^T, L in the lower triangle, upper set to 0.
// Returns 0, or -1 (and the failing column + 1 in *bad) when a pivot is not positive.
static int chol_lower(double* a, int n, int* bad) {
    for (int j = 0; j < n; ++j) {
        double* rj = a + (size_t) j * n;
        double d = rj[j];
        // d -= sum_k<j L[j][k]^2
        d4 acc = 0;
        int k = 0;
        for (; k + 4 <= j; k += 4) { const d4 v = ld4(rj + k); acc += v * v; }
        double s = acc.x + acc.y + acc.z + acc.w;
        for (; k < j; ++k) s += rj[k] * rj[k];
        d -= s;
        if (!(d > 0)) { if (bad) *bad = j + 1; return -1; }
        const double ljj = sqrt(d), inv = 1.0 / ljj;
        rj[j] = ljj;
        for (int i = j + 1; i < n; ++i) {
            double* ri = a + (size_t) i * n;
            d4 ac = 0;
            int q = 0;
            for (; q + 4 <= j; q += 4) {
                ac += ld4(ri + q) * ld4(rj + q);
            }
            double t = ac.x + ac.y + ac.z + ac.w;
            for (; q < j; ++q) t += ri[q] * rj[q];
            ri[j] = (ri[j] - t) * inv;
        }
    }
    for (int i = 0; i < n; ++i) memset(a + (size_t) i * n + i + 1, 0, sizeof(double) * (size_t) (n - i - 1));
    return 0;
}

// In-place inverse of a lower-triangular l (n x n): column by column, forward substitution on the identity.
static void tri_lower_inverse(double* l, int n) {
    double* inv = (double*) calloc((size_t) n * n, sizeof(double));
    double* col = (double*) malloc(sizeof(double) * (size_t) n);
    for (int j = 0; j < n; ++j) {
        // solve L x = e_j: x[i] = 0 for i < j
        memset(col, 0, sizeof(double) * (size_t) n);
        col[j] = 1.0 / l[(size_t) j * n + j];
        for (int i = j + 1; i < n; ++i) {
            const double* ri = l + (size_t) i * n;
            d4 ac = 0;
            int k = j;
            for (; k + 4 <= i; k += 4) ac += ld4(ri + k) * ld4(col + k);
            double s = ac.x + ac.y + ac.z + ac.w;
            for (; k < i; ++k) s += ri[k] * col[k];
            col[i] = -s / ri[i];
        }
        for (int i = j; i < n; ++i) inv[(size_t) i * n + j] = col[i];
    }
    memcpy(l, inv, sizeof(double) * (size_t) n * n);
    free(inv);
    free(col);
}

int nslm_gptq_factor(const double* h, int n, double damp, double* u, double* damp_used) {
    // dampen: H + damp * mean(diag H) * I (GPTQ's default damp = 0.01); dead inputs (zero diagonal) get the same floor
    double md = 0;
    for (int i = 0; i < n; ++i) md += h[(size_t) i * n + i];
    md /= n;
    const double add = damp * (md > 0 ? md : 1.0);
    if (damp_used) *damp_used = add;
    // u <- J (H + add I) J
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) u[(size_t) i * n + j] = h[(size_t) (n - 1 - i) * n + (n - 1 - j)] + (i == j ? add : 0.0);
    int bad = 0;
    if (chol_lower(u, n, &bad)) return -bad;
    tri_lower_inverse(u, n);   // M^-1 (lower)
    // U = J M^-1 J: reverse rows and columns (lower becomes upper)
    for (int i = 0; i < n / 2; ++i)
        for (int j = 0; j < n; ++j) {
            const double t = u[(size_t) i * n + j];
            u[(size_t) i * n + j] = u[(size_t) (n - 1 - i) * n + (n - 1 - j)];
            u[(size_t) (n - 1 - i) * n + (n - 1 - j)] = t;
        }
    if (n % 2) {   // the middle row reverses in place
        const int i = n / 2;
        for (int j = 0; j < n / 2; ++j) {
            const double t = u[(size_t) i * n + j];
            u[(size_t) i * n + j] = u[(size_t) i * n + (n - 1 - j)];
            u[(size_t) i * n + (n - 1 - j)] = t;
        }
    }
    return 0;
}

void nslm_upper8_inverse(const double* u, int ld, double t[64]) {
    // inverse of the 8 x 8 upper-triangular block at u (row stride ld): back substitution per column
    memset(t, 0, sizeof(double) * 64);
    for (int j = 0; j < 8; ++j) {
        t[j * 8 + j] = 1.0 / u[(size_t) j * ld + j];
        for (int i = j - 1; i >= 0; --i) {
            double s = 0;
            for (int k = i + 1; k <= j; ++k) s += u[(size_t) i * ld + k] * t[k * 8 + j];
            t[i * 8 + j] = -s / u[(size_t) i * ld + i];
        }
    }
}
