#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>

// Простая проверка ошибок CUDA, чтобы не ловить "тихие" баги
#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) \
                  << " | file: " << __FILE__ << " line: " << __LINE__ << "\n"; \
        std::exit(1); \
    } \
} while(0)

// GPU kernel: считаем сумму через глобальную память
// - каждый поток суммирует свою "полоску" элементов (stride по grid)
// - потом добавляет частичную сумму в один общий результат через atomicAdd
__global__ void sumGlobalAtomic(const float* __restrict__ d_arr, int n, float* d_sum)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    float local = 0.0f;
    for (int i = tid; i < n; i += stride) {
        local += d_arr[i]; // читаем из глобальной памяти
    }

    // общий итог хранится тоже в глобальной памяти
    atomicAdd(d_sum, local);
}

// CPU версия (последовательно)
double cpuSum(const std::vector<float>& a)
{
    double s = 0.0; // double, чтобы на CPU было аккуратнее по точности
    for (size_t i = 0; i < a.size(); ++i) {
        s += a[i];
    }
    return s;
}

int main()
{
    const int N = 100000; 
    std::vector<float> h_arr(N);

    // Заполним случайными числами, чтобы тест был "живой"
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < N; ++i) h_arr[i] = dist(rng);

    // CPU: сумма + замер времени
    auto cpu_t0 = std::chrono::high_resolution_clock::now();
    double cpu_result = cpuSum(h_arr);
    auto cpu_t1 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(cpu_t1 - cpu_t0).count();

    // GPU: память + сумма + замер времени
    float* d_arr = nullptr;
    float* d_sum = nullptr;

    CUDA_CHECK(cudaMalloc(&d_arr, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sum, sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_arr, h_arr.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_sum, 0, sizeof(float)));

    // Настройки запуска
    // Для N=100000 обычно хватает нескольких сотен блоков.
    // Берём 256 потоков в блоке — стандартный вариант.
    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    // Чтобы не запускать слишком много блоков (иногда только мешает), ограничим до разумного числа (например 1024).
    if (blocks > 1024) blocks = 1024;

    // CUDA Events — нормальный способ мерить время ядра на GPU
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    sumGlobalAtomic<<<blocks, threads>>>(d_arr, N, d_sum);
    CUDA_CHECK(cudaEventRecord(stop));

    // Проверяем, что ядро реально отработало без ошибок
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventSynchronize(stop));

    float gpu_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, start, stop));

    float gpu_result = 0.0f;
    CUDA_CHECK(cudaMemcpy(&gpu_result, d_sum, sizeof(float), cudaMemcpyDeviceToHost));

    // Сравнение результатов 
    // CPU у нас double, GPU float. Поэтому небольшая разница по округлению возможна.
    double diff = std::abs(cpu_result - static_cast<double>(gpu_result));

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "N = " << N << "\n\n";
    std::cout << "CPU sum  = " << cpu_result << "\n";
    std::cout << "GPU sum  = " << gpu_result << "\n";
    std::cout << "Abs diff = " << diff << "\n\n";

    std::cout << "CPU time (ms): " << cpu_ms << "\n";
    std::cout << "GPU kernel time (ms): " << gpu_ms << "\n";
    std::cout << "Grid: blocks=" << blocks << ", threads=" << threads << "\n";

    // Очистка
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUDA_CHECK(cudaFree(d_arr));
    CUDA_CHECK(cudaFree(d_sum));

    return 0;
}
