#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cstdint>

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = (call);                                            \
        if (err != cudaSuccess) {                                            \
            std::cerr << "CUDA error: " << cudaGetErrorString(err)           \
                      << " at " << __FILE__ << ":" << __LINE__ << "\n";      \
            std::exit(1);                                                    \
        }                                                                    \
    } while (0)

// Kernel: каждый блок считает частичную сумму в shared, затем один поток пишет сумму блока в global.
// Это соответствует "читаем из глобальной памяти", а сведение делаем внутри блока.
__global__ void sum_global_kernel(const int* d_in, long long* d_blockSums, int n) {
    extern __shared__ long long sdata[]; // shared для редукции

    int tid = threadIdx.x;
    int global_i = blockIdx.x * blockDim.x + tid;

    // Берём значение из глобальной памяти (или 0, если вышли за границу)
    long long x = 0;
    if (global_i < n) x = (long long)d_in[global_i];

    // Кладём в shared и редуцируем внутри блока
    sdata[tid] = x;
    __syncthreads();

    // Простая редукция: пополам, пока не останется 1 значение
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    // Поток 0 пишет сумму блока в глобальную память
    if (tid == 0) {
        d_blockSums[blockIdx.x] = sdata[0];
    }
}

static long long cpu_sum(const std::vector<int>& a) {
    long long s = 0;
    for (int v : a) s += (long long)v;
    return s;
}

int main() {
    const int N = 100000;          // размер массива по заданию
    const int BLOCK = 256;         // типичный размер блока
    const int GRID = (N + BLOCK - 1) / BLOCK; // сколько блоков нужно

    std::cout << "N=" << N << ", BLOCK=" << BLOCK << ", GRID=" << GRID << "\n";

    // Генерируем входные данные на CPU
    std::vector<int> h_in(N);
    std::mt19937 rng(12345);
    std::uniform_int_distribution<int> dist(0, 100);

    for (int i = 0; i < N; i++) h_in[i] = dist(rng);

    // CPU: считаем сумму и время 
    auto cpu_t0 = std::chrono::high_resolution_clock::now();
    long long cpu_res = cpu_sum(h_in);
    auto cpu_t1 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(cpu_t1 - cpu_t0).count();

    // GPU: выделяем память 
    int* d_in = nullptr;
    long long* d_blockSums = nullptr;

    CUDA_CHECK(cudaMalloc((void**)&d_in, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_blockSums, GRID * sizeof(long long)));

    // Копируем вход на GPU
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), N * sizeof(int), cudaMemcpyHostToDevice));

    // GPU: замер времени kernel через cudaEvent 
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    // shared memory на блок: BLOCK * sizeof(long long)
    sum_global_kernel<<<GRID, BLOCK, BLOCK * sizeof(long long)>>>(d_in, d_blockSums, N);

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    // Проверяем ошибки запуска kernel
    CUDA_CHECK(cudaGetLastError());

    float gpu_kernel_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_kernel_ms, start, stop));

    // Считываем суммы блоков назад и суммируем на CPU (финальный reduction) 
    std::vector<long long> h_blockSums(GRID);
    CUDA_CHECK(cudaMemcpy(h_blockSums.data(), d_blockSums, GRID * sizeof(long long), cudaMemcpyDeviceToHost));

    long long gpu_res = 0;
    for (int i = 0; i < GRID; i++) gpu_res += h_blockSums[i];

    // Вывод результатов 
    std::cout << "CPU sum: " << cpu_res << "\n";
    std::cout << "GPU sum: " << gpu_res << "\n";
    std::cout << "Match:   " << (cpu_res == gpu_res ? "YES" : "NO") << "\n\n";

    std::cout << "CPU time (ms):      " << cpu_ms << "\n";
    std::cout << "GPU kernel time(ms): " << gpu_kernel_ms << "\n";

    // Освобождаем ресурсы
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_blockSums));

    return 0;
}
