#ifndef _JUDGE_H_
#define _JUDGE_H_

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include <math.h>
#include <omp.h>

#define EPI     1e-6
#define MAXITER 20000
#define EPS     1e-6

void input(char filename[], int size, double A[][size], double b[]);

void check(int size, double A[][size], double x[], double b[], int iter);

void PCG(int size, double A[][size], double b[]);

void start_timer();

#endif