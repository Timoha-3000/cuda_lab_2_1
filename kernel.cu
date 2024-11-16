#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 1024   // Размер матриц
#define BLOCK_SIZE 16  // Размер блока

/// <summary>
/// Функция для проверки ошибок CUDA
/// </summary>
/// <param name="A">- Первый вектор</param>
/// <param name="B">- Второй вектор</param>
/// <param name="C">- Результирующий вектор</param>
/// <param name="n">- Размер векторов</param>
void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

/// <summary>
/// Перемножение матриц на CPU
/// </summary>
/// <param name="A">- Первый вектор</param>
/// <param name="B">- Второй вектор</param>
/// <param name="C">- Результирующий вектор</param>
/// <param name="n">- Размер векторов</param>
void matMulCPU(int* A, int* B, int* C, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            int sum = 0;
            for (int k = 0; k < n; k++) {
                sum += A[i * n + k] * B[k * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

/// <summary>
/// Ядро с использованием глобальной памяти
/// </summary>
/// <param name="A">- Первый вектор</param>
/// <param name="B">- Второй вектор</param>
/// <param name="C">- Результирующий вектор</param>
/// <param name="n">- Размер векторов</param>
__global__ void matMulGlobal(int* A, int* B, int* C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        int sum = 0;
        for (int k = 0; k < n; k++) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

/// <summary>
/// Ядро с использованием разделяемой памяти
/// </summary>
/// <param name="A">- Первый вектор</param>
/// <param name="B">- Второй вектор</param>
/// <param name="C">- Результирующий вектор</param>
/// <param name="n">- Размер векторов</param>
__global__ void matMulShared(int* A, int* B, int* C, int n) {
    // как бы, ну, разделяемая память
    __shared__ int sA[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int sB[BLOCK_SIZE][BLOCK_SIZE];

    // blockIdx.x;  // индексы блока
    // blockIdx.y;  //
    // threadIdx.x; // индексы нити внутри блока
    // threadIdx.y; //
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int sum = 0;

    for (int t = 0; t < (n + BLOCK_SIZE - 1) / BLOCK_SIZE; t++) {
        sA[ty][tx] = A[row * n + t * BLOCK_SIZE + tx];
        sB[ty][tx] = B[(t * BLOCK_SIZE + ty) * n + col];

        __syncthreads(); // Убедимся, что подматрицы полностью загружены

        for (int k = 0; k < BLOCK_SIZE; k++) {
            sum += sA[ty][k] * sB[k][tx];
        }
        __syncthreads(); // Убедимся, что подматрицы полностью загружены
    }

    if (row < n && col < n) {
        C[row * n + col] = sum;
    }
}

// Верификация результатов
void verifyResult(int* C1, int* C2, int n) {
    for (int i = 0; i < n * n; i++) {
        if (C1[i] != C2[i]) {
            printf("Error at index %d: %d != %d\n", i, C1[i], C2[i]);
            return;
        }
    }
    printf("Verification passed!\n");
}

/// <summary>
/// Второе задание лабороторной,
/// перемножение двух матриц в двумерной сетке.
/// </summary>
void multMatrixWith2DGridTask2() {
    int* h_A, * h_B, * h_C_cpu, * h_C_gpu_global, * h_C_gpu_shared;
    int* d_A, * d_B, * d_C;
    size_t size = N * N * sizeof(int);

    // Выделение памяти на хосте
    h_A = (int*)malloc(size);
    h_B = (int*)malloc(size);
    h_C_cpu = (int*)malloc(size);
    h_C_gpu_global = (int*)malloc(size);
    h_C_gpu_shared = (int*)malloc(size);

    // Инициализация случайных данных
    srand(time(NULL));
    for (int i = 0; i < N * N; i++) {
        h_A[i] = rand() % 10 + 1;
        h_B[i] = rand() % 10 + 1;
    }

    // Выделение памяти на устройстве
    checkCudaError(cudaMalloc((void**)&d_A, size), "cudaMalloc A");
    checkCudaError(cudaMalloc((void**)&d_B, size), "cudaMalloc B");
    checkCudaError(cudaMalloc((void**)&d_C, size), "cudaMalloc C");

    // Копирование данных с хоста на устройство
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Измерение времени работы на CPU
    clock_t start_cpu = clock();
    matMulCPU(h_A, h_B, h_C_cpu, N);
    clock_t end_cpu = clock();
    printf("CPU time: %f ms\n", 1000.0 * (end_cpu - start_cpu) / CLOCKS_PER_SEC);

    // Настройка сетки и блоков
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Глобальная память
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    matMulGlobal << <gridSize, blockSize >> > (d_A, d_B, d_C, N);
    cudaEventRecord(stop);
    cudaMemcpy(h_C_gpu_global, d_C, size, cudaMemcpyDeviceToHost);
    cudaEventSynchronize(stop);

    float time_global;
    cudaEventElapsedTime(&time_global, start, stop);
    printf("Global memory time: %f ms\n", time_global);

    // Верификация
    verifyResult(h_C_cpu, h_C_gpu_global, N);

    // Разделяемая память
    cudaEventRecord(start);
    matMulShared << <gridSize, blockSize >> > (d_A, d_B, d_C, N);
    cudaEventRecord(stop);
    cudaMemcpy(h_C_gpu_shared, d_C, size, cudaMemcpyDeviceToHost);
    cudaEventSynchronize(stop);

    float time_shared;
    cudaEventElapsedTime(&time_shared, start, stop);
    printf("Shared memory time: %f ms\n", time_shared);

    // Верификация
    verifyResult(h_C_cpu, h_C_gpu_shared, N);

    // Очистка памяти
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C_cpu);
    free(h_C_gpu_global);
    free(h_C_gpu_shared);
}

int main() {
    multMatrixWith2DGridTask2();

    return 0;
}
