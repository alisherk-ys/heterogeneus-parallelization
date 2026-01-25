#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>
#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#endif

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) \
                  << " | " << __FILE__ << ":" << __LINE__ << "\n"; \
        std::exit(1); \
    } \
} while(0)

// GPU kernel: y[i] = a*x[i] + b 
__global__ void affineKernel(const float* __restrict__ x,
                            float* __restrict__ y,
                            int n,
                            float a, float b)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + b;
}

// CPU обработка (можно с OpenMP) 
void cpuAffine(const float* x, float* y, int n, float a, float b)
{
    // Если OpenMP включён — распараллелим, если нет — будет обычный цикл
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (int i = 0; i < n; ++i) {
        y[i] = a * x[i] + b;
    }
}

// Проверка корректности 
float maxAbsDiff(const float* a, const float* b, int n)
{
    float mx = 0.0f;
    for (int i = 0; i < n; ++i) {
        float d = std::fabs(a[i] - b[i]);
        if (d > mx) mx = d;
    }
    return mx;
}

int main()
{
    const int N = 1'000'000;    // можно менять, но для гибрида лучше крупнее
    const float A = 1.2345f;
    const float B = -0.9876f;

    std::cout << std::fixed << std::setprecision(6);

    // Создаём pinned host memory 
    // pinned нужен, чтобы cudaMemcpyAsync реально работал асинхронно
    float *h_x = nullptr, *h_y_cpu = nullptr, *h_y_gpu = nullptr, *h_y_hybrid = nullptr;

    CUDA_CHECK(cudaMallocHost(&h_x,       N * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&h_y_cpu,   N * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&h_y_gpu,   N * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&h_y_hybrid,N * sizeof(float)));

    // Заполняем вход
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < N; ++i) h_x[i] = dist(rng);

    // CPU-only: весь массив на CPU 
    auto t0 = std::chrono::high_resolution_clock::now();
    cpuAffine(h_x, h_y_cpu, N, A, B);
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // GPU-only: весь массив на GPU 
    float *d_x = nullptr, *d_y = nullptr;
    CUDA_CHECK(cudaMalloc(&d_x, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y, N * sizeof(float)));

    int threads = 256;
    int blocks  = (N + threads - 1) / threads;

    cudaEvent_t gstart, gstop;
    CUDA_CHECK(cudaEventCreate(&gstart));
    CUDA_CHECK(cudaEventCreate(&gstop));

    auto tg0 = std::chrono::high_resolution_clock::now();

    CUDA_CHECK(cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaEventRecord(gstart));
    affineKernel<<<blocks, threads>>>(d_x, d_y, N, A, B);
    CUDA_CHECK(cudaEventRecord(gstop));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventSynchronize(gstop));

    float gpu_kernel_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_kernel_ms, gstart, gstop));

    CUDA_CHECK(cudaMemcpy(h_y_gpu, d_y, N * sizeof(float), cudaMemcpyDeviceToHost));

    auto tg1 = std::chrono::high_resolution_clock::now();
    double gpu_total_ms = std::chrono::duration<double, std::milli>(tg1 - tg0).count();

    // HYBRID: первая половина CPU, вторая GPU параллельно 
    const int N1 = N / 2;       // CPU часть
    const int N2 = N - N1;      // GPU часть (вторая половина)
    const float* h_x2 = h_x + N1;

    // На GPU выделим память только под вторую часть
    float *d_x2 = nullptr, *d_y2 = nullptr;
    CUDA_CHECK(cudaMalloc(&d_x2, N2 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y2, N2 * sizeof(float)));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    cudaEvent_t hstart, hstop;
    CUDA_CHECK(cudaEventCreate(&hstart));
    CUDA_CHECK(cudaEventCreate(&hstop));

    auto th0 = std::chrono::high_resolution_clock::now();

    // 1) Асинхронно отправляем вторую половину на GPU
    CUDA_CHECK(cudaMemcpyAsync(d_x2, h_x2, N2 * sizeof(float),
                               cudaMemcpyHostToDevice, stream));

    // 2) Пока копирование идёт — CPU считает первую половину
    cpuAffine(h_x, h_y_hybrid, N1, A, B);

    // 3) GPU считает вторую половину (в том же stream: после копирования)
    int blocks2 = (N2 + threads - 1) / threads;

    CUDA_CHECK(cudaEventRecord(hstart, stream));
    affineKernel<<<blocks2, threads, 0, stream>>>(d_x2, d_y2, N2, A, B);
    CUDA_CHECK(cudaEventRecord(hstop, stream));
    CUDA_CHECK(cudaGetLastError());

    // 4) Копируем обратно вторую половину в выход hybrid
    CUDA_CHECK(cudaMemcpyAsync(h_y_hybrid + N1, d_y2, N2 * sizeof(float),
                               cudaMemcpyDeviceToHost, stream));

    // 5) Ждём завершение GPU очереди
    CUDA_CHECK(cudaStreamSynchronize(stream));

    auto th1 = std::chrono::high_resolution_clock::now();
    double hybrid_total_ms = std::chrono::duration<double, std::milli>(th1 - th0).count();

    float hybrid_gpu_kernel_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&hybrid_gpu_kernel_ms, hstart, hstop));

    // Проверка корректности
    float err_gpu    = maxAbsDiff(h_y_cpu, h_y_gpu,    N);
    float err_hybrid = maxAbsDiff(h_y_cpu, h_y_hybrid, N);

    std::cout << "\nN = " << N << "\n";
    #ifdef _OPENMP
    std::cout << "OpenMP threads: " << omp_get_max_threads() << "\n";
    #else
    std::cout << "OpenMP: OFF (CPU будет последовательный)\n";
    #endif

    // Вывод результатов 
    std::cout << "\n--- Time (ms) ---\n";
    std::cout << "CPU-only total:    " << cpu_ms << "\n";
    std::cout << "GPU-only total:    " << gpu_total_ms << "   (includes H2D + kernel + D2H)\n";
    std::cout << "GPU-only kernel:   " << gpu_kernel_ms << "\n";
    std::cout << "Hybrid total:      " << hybrid_total_ms << "   (CPU+GPU overlap)\n";
    std::cout << "Hybrid GPU kernel: " << hybrid_gpu_kernel_ms << "\n";

    std::cout << "\n--- Correctness ---\n";
    std::cout << "Max abs error (CPU vs GPU):    " << err_gpu << "\n";
    std::cout << "Max abs error (CPU vs Hybrid): " << err_hybrid << "\n";

    // Освобождаем ресурсы
    CUDA_CHECK(cudaEventDestroy(gstart));
    CUDA_CHECK(cudaEventDestroy(gstop));
    CUDA_CHECK(cudaEventDestroy(hstart));
    CUDA_CHECK(cudaEventDestroy(hstop));
    CUDA_CHECK(cudaStreamDestroy(stream));

    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFree(d_x2));
    CUDA_CHECK(cudaFree(d_y2));

    CUDA_CHECK(cudaFreeHost(h_x));
    CUDA_CHECK(cudaFreeHost(h_y_cpu));
    CUDA_CHECK(cudaFreeHost(h_y_gpu));
    CUDA_CHECK(cudaFreeHost(h_y_hybrid));

    return 0;
}
