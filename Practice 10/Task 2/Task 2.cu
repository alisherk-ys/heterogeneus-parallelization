#include <cuda_runtime.h>   
#include <iostream>        
#include <vector>          
#include <iomanip>          
#include <cmath>           

// Макрос для проверки ошибок CUDA
#define CHECK(call) \
if ((call) != cudaSuccess) { \
    std::cerr << "CUDA error\n"; \
    exit(1); \
}

// 1. Коалесцированный доступ к памяти
// Потоки читают соседние элементы массива
__global__ void coalesced_kernel(const float* in, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        out[idx] = in[idx] * 2.0f;
}

// 2. Некоалесцированный доступ
// Потоки обращаются к памяти с шагом (idx * 2)
__global__ void noncoalesced_kernel(const float* in, float* out, int n) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    if (idx < n)
        out[idx] = in[idx] * 2.0f;
}

// 3. Использование shared memory
// Данные сначала копируются в shared-память
__global__ void shared_kernel(const float* in, float* out, int n) {
    __shared__ float tile[256];  // разделяемая память блока

    int gid = blockIdx.x * blockDim.x + threadIdx.x; // глобальный индекс
    int tid = threadIdx.x;                            // локальный индекс в блоке

    if (gid < n)
        tile[tid] = in[gid];  // загрузка в shared memory
    __syncthreads();          // синхронизация потоков блока

    if (gid < n)
        out[gid] = tile[tid] * 2.0f;
}

// Функция замера времени выполнения ядра
float measure(void(*kernel)(const float*, float*, int),
              const float* d_in, float* d_out,
              int n, dim3 grid, dim3 block)
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Прогрев (warm-up)
    kernel<<<grid, block>>>(d_in, d_out, n);

    // Замер времени
    cudaEventRecord(start);
    kernel<<<grid, block>>>(d_in, d_out, n);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    return ms;
}

int main() {
    const int N = 1'000'000;   // размер массива
    const int BLOCK = 256;     // размер блока
    dim3 block(BLOCK);
    dim3 grid((N + BLOCK - 1) / BLOCK);

    // Хост-массивы
    std::vector<float> h_in(N, 1.0f), h_out(N);

    // Указатели на device-память
    float *d_in, *d_out;
    CHECK(cudaMalloc(&d_in, N * sizeof(float)));
    CHECK(cudaMalloc(&d_out, N * sizeof(float)));
    CHECK(cudaMemcpy(d_in, h_in.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    // Замеры времени для разных ядер
    float t1 = measure(coalesced_kernel, d_in, d_out, N, grid, block);
    float t2 = measure(noncoalesced_kernel, d_in, d_out, N, grid, block);
    float t3 = measure(shared_kernel, d_in, d_out, N, grid, block);

    // Вывод результатов
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Coalesced access time (ms)    = " << t1 << "\n";
    std::cout << "Non-coalesced access time (ms)= " << t2 << "\n";
    std::cout << "Shared memory time (ms)       = " << t3 << "\n";

    // Освобождение памяти
    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}
