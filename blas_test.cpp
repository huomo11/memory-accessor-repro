#include <cblas.h>
#include <iostream>

int main() {
    // Row-major 2x2 matrix:
    // [1 2
    //  3 4]
    double A[4] = {1.0, 2.0, 3.0, 4.0};
    double x[2] = {1.0, 1.0};
    double y[2] = {0.0, 0.0};

    cblas_dgemv(
        CblasRowMajor,
        CblasNoTrans,
        2, 2,
        1.0,
        A, 2,
        x, 1,
        0.0,
        y, 1
    );

    std::cout << y[0] << " " << y[1] << std::endl;
    return 0;
}