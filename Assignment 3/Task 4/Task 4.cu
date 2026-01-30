#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include <cmath>

// Макрос для проверки ошибок CUDA-вызовов.
#define CUDA_CHECK(call) do {                                      \
    cudaError_t err = (call);                                      \
    if (err != cudaSuccess) {                                      \
        std::cerr << "CUDA error: " << cudaGetErrorString(err)     \
                  << " at " << __FILE__ << ":" << __LINE__         \
                  << std::endl;                                    \
        std::exit(1);                                              \
    }                                                              \
} while(0)

// Ядро: поэлементное сложение A и B -> C.
__global__ void vec_add(const float* __restrict__ A,
                        const float* __restrict__ B,
                        float* __restrict__ C,
                        int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) C[idx] = A[idx] + B[idx];
}

// Замер среднего времени vec_add для заданного размера блока.
// grid вычисляется автоматически из N и block.
float timeVecAdd(const float* dA, const float* dB, float* dC, int n,
                 int block, int iters)
{
    int grid = (n + block - 1) / block;

    // Прогрев (первый запуск) + синхронизация
    vec_add<<<grid, block>>>(dA, dB, dC, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // CUDA events для замера времени
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) {
        vec_add<<<grid, block>>>(dA, dB, dC, n);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return ms / iters; // среднее время одного запуска
}

int main() {
    const int N = 1'000'000;   // размер массивов
    const int iters = 300;     // повторы для замера

    // Набор кандидатов на "оптимальный" размер блока
    const int candidates[] = {64, 128, 192, 256, 320, 384, 512, 768, 1024};
    const int numCand = sizeof(candidates) / sizeof(candidates[0]);

    // "Неоптимальная" конфигурация (выбрана намеренно как baseline)
    const int unopt_block = 1024;

    // Данные на CPU: A, B, C и эталон Ref
    std::vector<float> hA(N), hB(N), hC(N), hRef(N);

    // Заполнение случайными числами и расчёт эталона на CPU
    std::mt19937 gen(123);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < N; ++i) {
        hA[i] = dist(gen);
        hB[i] = dist(gen);
        hRef[i] = hA[i] + hB[i];
    }

    // Память на GPU
    float *dA=nullptr, *dB=nullptr, *dC=nullptr;
    CUDA_CHECK(cudaMalloc(&dA, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dB, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dC, N * sizeof(float)));

    // Копирование A и B на GPU
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    std::cout << "N = " << N << ", iters = " << iters << "\n\n";

    // 1) Замер "неоптимальной" конфигурации
    float unopt_ms = timeVecAdd(dA, dB, dC, N, unopt_block, iters);
    int unopt_grid = (N + unopt_block - 1) / unopt_block;

    std::cout << "Unoptimized config:\n";
    std::cout << "  Block = " << unopt_block << ", Grid = " << unopt_grid
              << ", Avg kernel time (ms) = " << std::fixed << std::setprecision(6) << unopt_ms << "\n\n";

    // 2) Поиск лучшего размера блока среди candidates
    float best_ms = 1e9f;
    int best_block = -1;
    int best_grid = -1;

    std::cout << std::left
              << std::setw(10) << "Block"
              << std::setw(10) << "Grid"
              << std::setw(20) << "Avg kernel (ms)"
              << "\n";
    std::cout << std::string(40, '-') << "\n";

    for (int i = 0; i < numCand; ++i) {
        int block = candidates[i];
        int grid  = (N + block - 1) / block;

        float ms = timeVecAdd(dA, dB, dC, N, block, iters);

        std::cout << std::left
                  << std::setw(10) << block
                  << std::setw(10) << grid
                  << std::setw(20) << std::fixed << std::setprecision(6) << ms
                  << "\n";

        // Сохраняем лучшую конфигурацию (минимальное время)
        if (ms < best_ms) {
            best_ms = ms;
            best_block = block;
            best_grid = grid;
        }
    }

    // 3) Проверка корректности: запуск с лучшей конфигурацией и сравнение с эталоном
    vec_add<<<best_grid, best_block>>>(dA, dB, dC, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(hC.data(), dC, N * sizeof(float), cudaMemcpyDeviceToHost));

    float maxErr = 0.0f;
    for (int i = 0; i < N; ++i) {
        float err = std::fabs(hC[i] - hRef[i]);
        if (err > maxErr) maxErr = err;
    }
    bool ok = (maxErr <= 1e-4f);

    // Итоги: лучшая конфигурация и сравнение с baseline
    std::cout << "\nOptimized config (best among tested):\n";
    std::cout << "  Block = " << best_block << ", Grid = " << best_grid
              << ", Avg kernel time (ms) = " << best_ms << "\n";

    std::cout << "\nComparison:\n";
    std::cout << "  Unoptimized time (ms) = " << unopt_ms << "\n";
    std::cout << "  Optimized time (ms)   = " << best_ms << "\n";
    std::cout << "  Speedup (unopt/opt)   = " << (unopt_ms / best_ms) << "x\n";

    std::cout << "\nCorrect = " << (ok ? "YES" : "NO") << "\n";
    std::cout << "Max abs error = " << maxErr << "\n";

    // Освобождение памяти GPU
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    return 0;
}
