#include "judge.h"

double dot(double a[], double b[], int size){
    double dot_sum = 0;
    for (int i = 0; i < size; ++i) {
        dot_sum += a[i] * b[i];
    }
    return dot_sum;
}

void MatrixMulVec(int size, double A[][size], double x[], double b[]) {
    for (int i = 0; i < size; ++i) {
        b[i] = 0;
        for (int j = 0; j < size; ++j) {
            b[i] += A[i][j] * x[j];
        }
    }
}

void PCG(int size, double A[][size], double b[]) {

    start_timer();

    double r[size], x[size], z[size], p[size];
    double alpha, beta, gamma, r_dot_z, A_dot_p[size];

    /* initialization */
    for (int i = 0; i < size; ++i){
        x[i] = 0;
        r[i] = b[i];
        z[i] = r[i] / A[i][i];
        p[i] = z[i];
    }

    /* solve */
    int iter = 0;
    double loss = 0;
    r_dot_z = dot(r, z, size);
    do {            
        /* A * p_k */
        MatrixMulVec(size, A, p, A_dot_p);

        /* alpha */
        alpha = r_dot_z / dot(p, A_dot_p, size);

        /* x */
        for (int i = 0; i < size; ++i) {
            x[i] += alpha * p[i];
        }

        /* r & loss */
        int quit = 1;
        loss = 0;
        for (int i = 0; i < size; ++i) {
            r[i] = r[i] - alpha * A_dot_p[i];
            if (r[i]) quit = 0;
            loss += fabs(r[i]);
        }
        if (quit) break;

        /* z */
        for (int i = 0; i < size; ++i) {
            z[i] = r[i] / A[i][i];
        }

        /* beta */
        double temp = dot(z, r, size);
        beta = temp / r_dot_z;
        r_dot_z = temp;

        /* p */
        for (int i = 0; i < size; ++i) {
            p[i] = z[i] + beta * p[i];
        }
    }
    while (++iter < MAXITER && loss > EPI);

    check(size, A, x, b, iter);
}