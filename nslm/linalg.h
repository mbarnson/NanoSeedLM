// nslm/linalg.h - dense linear algebra for GPTQ-style error feedback (S-01 S3), double precision, C99.
#pragma once

// U with H^-1 = U^T U (upper triangular, positive diagonal) for the dampened Hessian H + damp * mean(diag H) * I.
// h, u: n x n row-major (may not alias).  *damp_used (optional) gets the absolute damping.  Returns 0, or -(column + 1)
// when the dampened matrix is not positive definite.
int nslm_gptq_factor(const double* h, int n, double damp, double* u, double* damp_used);

// t = the inverse of the 8 x 8 upper-triangular block at u (row stride ld), row-major 8 x 8 (upper triangular).
void nslm_upper8_inverse(const double* u, int ld, double t[64]);
