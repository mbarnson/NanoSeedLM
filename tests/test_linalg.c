// tests/test_linalg.c - nslm/lib_linalg.c: the GPTQ factor U (H^-1 = U^T U) and the 8 x 8 block inverse.
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "linalg.h"

static int g_fail;

static void check(const char* what, double err, double tol) {
    const int ok = err <= tol;
    printf("%-60s %s (%.2e)\n", what, ok ? "ok" : "FAIL", err);
    g_fail += !ok;
}

int main(void) {
    srand(7);
    const int ns[3] = {16, 40, 203};
    for (int ni = 0; ni < 3; ++ni) {
        const int n = ns[ni], m = 3 * n;
        // H = X^T X for random X (m x n): SPD, with a few strongly correlated columns like real activations
        double* x = malloc(sizeof(double) * (size_t) m * n), *h = calloc((size_t) n * n, sizeof(double));
        for (int i = 0; i < m * n; ++i) x[i] = rand() / (double) RAND_MAX - 0.5;
        for (int r = 0; r < m; ++r) x[(size_t) r * n + 1] = x[(size_t) r * n] * 0.99 + x[(size_t) r * n + 1] * 0.01;
        for (int r = 0; r < m; ++r)
            for (int i = 0; i < n; ++i)
                for (int j = 0; j < n; ++j) h[(size_t) i * n + j] += x[(size_t) r * n + i] * x[(size_t) r * n + j];
        double* u = malloc(sizeof(double) * (size_t) n * n);
        double damp = 0;
        const int rc = nslm_gptq_factor(h, n, 0.01, u, &damp);
        char w[96];
        snprintf(w, sizeof w, "n=%d factor succeeds", n);
        check(w, rc ? 1.0 : 0.0, 0.0);
        // upper triangular, positive diagonal
        double low = 0, mind = INFINITY;
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < i; ++j) low = fmax(low, fabs(u[(size_t) i * n + j]));
            mind = fmin(mind, u[(size_t) i * n + i]);
        }
        snprintf(w, sizeof w, "n=%d U upper triangular", n);
        check(w, low, 0.0);
        snprintf(w, sizeof w, "n=%d U diagonal positive", n);
        check(w, mind > 0 ? 0.0 : 1.0, 0.0);
        // (U^T U)(H + damp I) = I
        double worst = 0;
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j) {
                double s = 0;
                for (int k = 0; k < n; ++k) {
                    double ut = 0;   // (U^T U)[i][k]
                    for (int q = 0; q <= (i < k ? i : k); ++q) ut += u[(size_t) q * n + i] * u[(size_t) q * n + k];
                    s += ut * (h[(size_t) k * n + j] + (k == j ? damp : 0.0));
                }
                worst = fmax(worst, fabs(s - (i == j ? 1.0 : 0.0)));
            }
        snprintf(w, sizeof w, "n=%d (U^T U)(H + damp) = I", n);
        check(w, worst, 1e-9);
        // 8 x 8 diagonal block inverses
        double wb = 0;
        for (int b = 0; b + 8 <= n; b += 8) {
            double t[64];
            nslm_upper8_inverse(u + (size_t) b * n + b, n, t);
            for (int i = 0; i < 8; ++i)
                for (int j = 0; j < 8; ++j) {
                    double s = 0;
                    for (int k = 0; k < 8; ++k) s += u[(size_t) (b + i) * n + b + k] * t[k * 8 + j];
                    wb = fmax(wb, fabs(s - (i == j ? 1.0 : 0.0)));
                }
        }
        snprintf(w, sizeof w, "n=%d U_BB * inverse = I (every 8 x 8 block)", n);
        check(w, wb, 1e-12);
        free(x); free(h); free(u);
    }
    printf("test_linalg: %s\n", g_fail ? "FAIL" : "PASS");
    return g_fail ? 1 : 0;
}
