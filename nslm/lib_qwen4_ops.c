// nslm/lib_qwen4_ops.c - the qwen4_exp operations of one token (nslm/qwen4_ops.h).
#include "qwen4_ops.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

static double sigm(double x) { return 1.0 / (1.0 + exp(-x)); }

void qwen4_norm1p(const float* x, int len, int g, const float* w, float eps, float* y) {
    for (int a = 0; a < len; a += g) {
        double ss = 0;
        for (int k = 0; k < g; ++k) ss += (double) x[a + k] * x[a + k];
        const double s = 1.0 / sqrt(ss / g + eps);
        for (int k = 0; k < g; ++k) y[a + k] = (float) (x[a + k] * s * (1.0 + w[a + k]));
    }
}

size_t qwen4_gdn_conv_len(const Qwen4Cfg* c) {
    return (size_t) (c->lin_conv - 1) * (size_t) (2 * c->lin_hk * c->lin_dk + c->lin_hv * c->lin_dv);
}
size_t qwen4_gdn_state_len(const Qwen4Cfg* c) { return (size_t) c->lin_hv * c->lin_dk * c->lin_dv; }

void qwen4_gdn_token(const Qwen4Cfg* c, const Qwen4GdnW* w, const float* qkv, const float* z, const float* a, const float* b,
                     float* conv, float* S, float* y) {
    const int Hk = c->lin_hk, Hv = c->lin_hv, Dk = c->lin_dk, Dv = c->lin_dv, K = c->lin_conv;
    const int kd = Hk * Dk, cd = 2 * kd + Hv * Dv, rep = Hv / Hk;
    float* cv = (float*) malloc(sizeof(float) * (size_t) cd);
    for (int ch = 0; ch < cd; ++ch) {   // depthwise causal conv over the last K inputs, then silu
        double acc = (double) w->conv_w[(size_t) ch * K + K - 1] * qkv[ch];
        for (int k = 0; k < K - 1; ++k) acc += (double) w->conv_w[(size_t) ch * K + k] * conv[(size_t) k * cd + ch];
        cv[ch] = (float) (acc * sigm(acc));
    }
    if (K > 1) {
        memmove(conv, conv + cd, sizeof(float) * (size_t) (K - 2) * cd);
        memcpy(conv + (size_t) (K - 2) * cd, qkv, sizeof(float) * (size_t) cd);
    }
    for (int h = 0; h < 2 * Hk; ++h) {   // L2-normalise q and k heads; q also scaled by dk^-0.5
        float* v = cv + (size_t) h * Dk;
        double ss = 0;
        for (int j = 0; j < Dk; ++j) ss += (double) v[j] * v[j];
        const double s = 1.0 / sqrt(ss + 1e-6) * (h < Hk ? 1.0 / sqrt((double) Dk) : 1.0);
        for (int j = 0; j < Dk; ++j) v[j] = (float) (v[j] * s);
    }
    for (int hv = 0; hv < Hv; ++hv) {
        const float* q = cv + (size_t) (hv / rep) * Dk;
        const float* k = cv + kd + (size_t) (hv / rep) * Dk;
        const float* v = cv + 2 * kd + (size_t) hv * Dv;
        const double ap = (double) a[hv] + w->dt_bias[hv], sp = ap > 20 ? ap : log1p(exp(ap));
        const double g = exp(-exp((double) w->A_log[hv]) * sp), beta = sigm(b[hv]);
        float* St = S + (size_t) hv * Dk * Dv;   // [dk][dv]
        float* yo = y + (size_t) hv * Dv;
        double ss = 0;
        for (int i = 0; i < Dv; ++i) {
            double kv = 0, yq = 0;
            for (int j = 0; j < Dk; ++j) kv += (double) St[(size_t) j * Dv + i] * k[j];
            const double delta = (v[i] - g * kv) * beta;
            for (int j = 0; j < Dk; ++j) {
                const float s = (float) (g * St[(size_t) j * Dv + i] + delta * k[j]);
                St[(size_t) j * Dv + i] = s;
                yq += (double) s * q[j];
            }
            yo[i] = (float) yq;
            ss += yq * yq;
        }
        const double s = 1.0 / sqrt(ss / Dv + c->eps);   // RMSNorm (plain weight), gated by sigmoid(z)
        for (int i = 0; i < Dv; ++i) yo[i] = (float) (yo[i] * s * w->norm_w[i] * sigm(z[(size_t) hv * Dv + i]));
    }
    free(cv);
}

void qwen4_hc_act(const Qwen4Cfg* c, float* lo) {
    for (int i = 0; i < c->hc_rank; ++i) { const double x = lo[i] / (double) c->hc; lo[i] = (float) (x * sigm(x)); }
}
void qwen4_hc_mix(const Qwen4Cfg* c, const float* normed, const float* up, float* mixed) {
    for (int i = 0; i < c->d; ++i) {
        double m = 0;
        for (int s = 0; s < c->hc; ++s) m += sigm(up[s * c->d + i]) * normed[s * c->d + i];
        mixed[i] = (float) (m / c->hc);
    }
}
void qwen4_hc_gate(const Qwen4Cfg* c, float* inj) {
    for (int s = 0; s < c->hc; ++s) inj[s] = (float) (2.0 * sigm(inj[s] / (double) c->hc));
}
void qwen4_hc_add(const Qwen4Cfg* c, float* H, const float* branch, const float* inj) {
    for (int s = 0; s < c->hc; ++s)
        for (int i = 0; i < c->d; ++i) H[s * c->d + i] += branch[i] * inj[s];
}
