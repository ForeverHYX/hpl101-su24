#include "judge.h"

double start, end, total_time;
struct timeval tv;

void input(char filename[], int size, double A[][size], double b[]) {

    FILE *input = fopen(filename, "rb");
    fseek(input, 4, SEEK_SET);

    /* read A */
    float Af[size][size], bf[size];
    for (int i = 0; i < size; ++i) {
        fread(&Af[i][i], sizeof(float), size - i, input);
        for (int j = i; j < size; ++j) {
            Af[j][i] = Af[i][j];
        }
        for (int j = 0; j < size; ++j) {
            A[i][j] = Af[i][j];
        }
    }

    /* read b */
    fread(bf, sizeof(float), size, input);
    for (int i = 0; i < size; ++i) {
        b[i] = bf[i];
    }

    printf("input complete.\n");
}

void start_timer() {
    gettimeofday(&tv, NULL);
    start = (double)(tv.tv_sec) + (double)(tv.tv_usec) * 1e-6;
}

void check(int size, double A[][size], double x[], double b[], int iter) {

    /* timer end */
    gettimeofday(&tv, NULL);
    end = (double)(tv.tv_sec) + (double)(tv.tv_usec) * 1e-6;

    /* check correctness */
    for (int i = 0; i < size; ++i) {
        double temp = 0;
        for (int j = 0; j < size; ++j) {
            temp += A[i][j] * x[j];
        }
        if (fabs(temp - b[i]) > EPI) {
            printf("Validation failed at x[%d]: (Ax)[%d] = %.7lf, b[%d] = %.7lf\n",\
            i, i, temp, i, b[i]);
            exit(1);
        }
    }

    printf("Validation passed. total time = %.4lfs, iteration = %d\n", end - start, iter);
}