#pragma once

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    CblasRowMajor = 101,
    CblasColMajor = 102
} CBLAS_ORDER;

typedef CBLAS_ORDER CBLAS_LAYOUT;

typedef enum {
    CblasNoTrans = 111,
    CblasTrans = 112,
    CblasConjTrans = 113
} CBLAS_TRANSPOSE;

typedef enum {
    CblasUpper = 121,
    CblasLower = 122
} CBLAS_UPLO;

typedef enum {
    CblasNonUnit = 131,
    CblasUnit = 132
} CBLAS_DIAG;

void cblas_dtrsv(
    const CBLAS_LAYOUT Layout,
    const CBLAS_UPLO Uplo,
    const CBLAS_TRANSPOSE TransA,
    const CBLAS_DIAG Diag,
    const int N,
    const double *A,
    const int lda,
    double *X,
    const int incX
);

void cblas_strsv(
    const CBLAS_LAYOUT Layout,
    const CBLAS_UPLO Uplo,
    const CBLAS_TRANSPOSE TransA,
    const CBLAS_DIAG Diag,
    const int N,
    const float *A,
    const int lda,
    float *X,
    const int incX
);

void cblas_dgemv(
    const CBLAS_LAYOUT Layout,
    const CBLAS_TRANSPOSE TransA,
    const int M,
    const int N,
    const double alpha,
    const double *A,
    const int lda,
    const double *X,
    const int incX,
    const double beta,
    double *Y,
    const int incY
);

void cblas_sgemv(
    const CBLAS_LAYOUT Layout,
    const CBLAS_TRANSPOSE TransA,
    const int M,
    const int N,
    const float alpha,
    const float *A,
    const int lda,
    const float *X,
    const int incX,
    const float beta,
    float *Y,
    const int incY
);

#ifdef __cplusplus
}
#endif
