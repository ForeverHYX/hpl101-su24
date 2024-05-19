#include <cuda.h>
#include <cublas_v2.h>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <omp.h>

const int block_size = 16;
const int size = 10001;     // Matrix Size (size * size)
const int iter = 2;     // Number of iterations

#define a(_x, _y) a[(_x) * size + (_y)]
#define b(_x, _y) b[(_x) * size + (_y)]
#define result(_x, _y) result[(_x) * size + (_y)]
#define CUDA_CALL(func)                                               \
  {                                                                   \
    cudaError_t e = (func);                                           \
    if (!(e == cudaSuccess || e == cudaErrorCudartUnloading))         \
    {                                                                 \
      fprintf(stderr, "CUDA: %s:%d: error: %s\n", __FILE__, __LINE__, \
              cudaGetErrorString(e));                                 \
      abort();                                                        \
    }                                                                 \
  }
#define CUBLAS_CALL(func)                                             \
  {                                                                   \
    cublasStatus_t e = (func);                                        \
    if (!(e == CUBLAS_STATUS_SUCCESS))                                \
    {                                                                 \
      fprintf(stderr, "CUBLAS: %s:%d: error: %d\n", __FILE__, __LINE__, \
              e);                                 \
      abort();                                                        \
    }                                                                 \
  }

/// \brief Simply generate a random matrix.
void Generate(double *const a) {
  srand(time(NULL));
  // Matrix row.
#pragma omp parallel for
  for (int i = 0; i < size; ++i) {
    // Matrix column.
    for (int j = 0; j < size; ++j) {
      // Matrix element.
      a(i, j) = rand() % 100 / 100.0f;
    }
  }
}

/// \brief Check the correctness of the result and compare performace by using Cublas.
void CublasImplete(const double *__restrict__ a,
                   const double *__restrict__ b,
                   double *__restrict__ result,
                   cudaEvent_t *start_cublas, cudaEvent_t *stop_cublas) {
  double *a_kernel_1, *a_kernel_2, *b_kernel, *result_kernel;
  CUDA_CALL(cudaMalloc(&a_kernel_1, size * size * sizeof(double)));
  CUDA_CALL(cudaMemcpy(a_kernel_1, a, size * size * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMalloc(&a_kernel_2, size * size * sizeof(double)));
  CUDA_CALL(cudaMemcpy(a_kernel_2, a_kernel_1, size * size * sizeof(double), cudaMemcpyDeviceToDevice));
  CUDA_CALL(cudaMalloc(&b_kernel, size * size * sizeof(double)));
  CUDA_CALL(cudaMemcpy(b_kernel, b, size * size * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMalloc(&result_kernel, size * size * sizeof(double)));

  cudaEventRecord(*start_cublas);
  // Use cublasDgeam to (A + (k - 1) * B) + B -> A + k * B.
  cublasHandle_t handle;
  CUBLAS_CALL(cublasCreate(&handle));
  double alpha = 1.0f;
  double betageam = 1.0f;
  double betagemm = 0.0f;
  for (int i = 0; i < iter; ++i) {
    CUBLAS_CALL(cublasDgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, size, size, &alpha, a_kernel_2, size, &betageam, b_kernel, size, a_kernel_2, size));
    
    CUBLAS_CALL(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, size, size, size, &alpha, a_kernel_2, size, a_kernel_1, size, &betagemm, result_kernel, size));
    double *tmp = a_kernel_1;
    a_kernel_1 = result_kernel;
    result_kernel = tmp;
  }

  cudaEventRecord(*stop_cublas);
  cudaEventSynchronize(*stop_cublas);

  CUBLAS_CALL(cublasGetMatrix(size, size, sizeof(double), a_kernel_1, size, result, size));
  CUBLAS_CALL(cublasDestroy(handle));

  return;
}

/// \brief Check the correctness of the result.
void Verify(const double *const result,
            const double *const result_cublas) {
  bool correct = true;
  for (int i = 0; i < size * size; ++i) {
    // check if there is inf
    if (result[i] == INFINITY || result[i] == -INFINITY) {
      std::cout << "\x1b[31m"
                   "Wrong Answer"
                   "\x1b[0m"
                   " at ("
                << i / size << ", " << i % size << "): ";
      std::cout << "expected " << std::setprecision(6) << result_cublas[i]
                << ", got " << result[i]
                << std::endl;
      correct = false;
      break;
    }
    if (fabs(result[i]) < 1e-6 || fabs(result_cublas[i]) < 1e-6) {
      std::cout << "\x1b[31m"
                   "Wrong Answer"
                   "\x1b[0m"
                   " at ("
                << i / size << ", " << i % size << "): ";
      std::cout << "expected " << std::setprecision(6) << result_cublas[i]
                << ", got " << result[i]
                << std::endl;
      correct = false;
      break;
    }
    double error = fabs(result[i] - result_cublas[i]) / result_cublas[i];
    if (error > 1e-6) {
      correct = false;
      std::cout << "\x1b[31m"
                   "Wrong Answer"
                   "\x1b[0m"
                   " at ("
                << i / size << ", " << i % size << "): ";
      std::cout << "expected " << std::setprecision(6) << result_cublas[i]
                << ", got " << result[i]
                << std::endl;
      break;
    }
  }
  if (correct) {
    std::cout << "\x1b[32m"
                "Correct"
                "\x1b[0m"
              << std::endl;
  }
  return;
}

/// \brief Let A to be A + B.
__global__ void AdderCudaKernel(double *__restrict__ a,
                                  const double *__restrict__ b)
{
  const int i = blockIdx.x * block_size + threadIdx.x;
  const int j = blockIdx.y * block_size + threadIdx.y;
  if (i < size && j < size) {
    a(i, j) += b(i, j);
  }
}

/// \brief Do Matrix Multiplication on GPU.
__global__ void MultipleCudaKernel(const double *__restrict__ a, 
                                     const double *__restrict__ b, 
                                     double *__restrict__ result) 
{
  const int i = blockIdx.x * block_size + threadIdx.x;
  const int j = blockIdx.y * block_size + threadIdx.y;
  if (i < size && j < size) {
    result(i, j) = 0;
    for (int k = 0; k < size; ++k) {
      result(i, j) += a(i, k) * b(k, j);
    }
  }
}

// Naive implementation, only for testing correctness and precision
void MultipleCuda(const double *const a, const double *const b, double *const result,
                   cudaEvent_t *start_e, cudaEvent_t *stop_e) 
{
  double *a_kernel, *b_kernel, *copy_kernel, *result_kernel;
  CUDA_CALL(cudaMalloc(&a_kernel, size * size * sizeof(double)));
  CUDA_CALL(cudaMemcpy(a_kernel, a, size * size * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMalloc(&b_kernel, size * size * sizeof(double)));
  CUDA_CALL(cudaMemcpy(b_kernel, b, size * size * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMalloc(&copy_kernel, size * size * sizeof(double)));
  CUDA_CALL(cudaMemcpy(copy_kernel, a_kernel, size * size * sizeof(double), cudaMemcpyDeviceToDevice));
  CUDA_CALL(cudaMalloc(&result_kernel, size * size * sizeof(double)));
  
  // Start Timer.
  cudaEventRecord(*start_e);
  
  // Run Matrix Multiplication.
  // Parameters to be set:
  dim3 grid((size + block_size - 1) / block_size,
            (size + block_size - 1) / block_size);
  dim3 block(block_size, block_size);

  // Calculate \Prod_{k=0}^{n} (A + k * B).
  for (int i = 0; i < iter; ++i) {
    // @note: you can also use CUDA API to launch a cuda kernel function,
    // __host__ cudaError_t cudaLaunchKernel;
    // Perform (A + (k - 1) * B) + B -> A + k * B.
    AdderCudaKernel<<<grid, block>>>(copy_kernel, b_kernel);
    CUDA_CALL(cudaDeviceSynchronize());
    // Perform A * B -> Result.
    MultipleCudaKernel<<<grid, block>>>(a_kernel, copy_kernel, result_kernel);
    CUDA_CALL(cudaDeviceSynchronize());

    // Swap pointers between A and Result.
    double *tmp = a_kernel;
    a_kernel = result_kernel;
    result_kernel = tmp;
  }

  // Stop Timer
  cudaEventRecord(*stop_e);
  cudaEventSynchronize(*stop_e);

  // At the end of the loop, the result is in a_kernel.
  CUDA_CALL(cudaMemcpy(result, a_kernel, size * size * sizeof(double), cudaMemcpyDeviceToHost));
  cudaFree(a_kernel);
  cudaFree(b_kernel);
  cudaFree(copy_kernel);
  cudaFree(result_kernel);
}

int main() {
  auto a = new double[size * size];
  auto b = new double[size * size];
  auto result = new double[size * size];
  auto result_cublas = new double[size * size];
  std::cout << "Generating input matrices... \n";
  Generate(a);
  Generate(b);

  cudaEvent_t start_e, stop_e;
  cudaEventCreate(&start_e);
  cudaEventCreate(&stop_e);

  // Perform Matrix Multiplication on GPU.
  std::cout << "Custom Matrix Multiplication on GPU... \n";
  MultipleCuda(a, b, result, &start_e, &stop_e);

  cudaEvent_t start_cublas, stop_cublas;
  cudaEventCreate(&start_cublas);
  cudaEventCreate(&stop_cublas);
  std::cout << "cuBLAS Matrix Multiplication on GPU... \n";
  CublasImplete(a, b, result_cublas, &start_cublas, &stop_cublas);

  std::cout << "Verifying... \n";
  // Verify the result.
  Verify(result, result_cublas);

  // Calculate to evaluate performance.
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start_e, stop_e);
  std::cout << "Custom: " << milliseconds << " milliseconds" << std::endl;
  cudaEventElapsedTime(&milliseconds, start_cublas, stop_cublas);
  std::cout << "cuBLAS: " << milliseconds << " milliseconds" << std::endl;
  cudaEventDestroy(start_e);
  cudaEventDestroy(stop_e);
  cudaEventDestroy(start_cublas);
  cudaEventDestroy(stop_cublas);

  // Delete allocated memory.
  delete[] a;
  delete[] b;
  delete[] result;
  return 0;
}