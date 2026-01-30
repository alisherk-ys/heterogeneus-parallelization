#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include <cmath>

// Макрос проверки ошибок CUDA (для всех вызовов API).
#define CUDA_CHECK(call) do {                                      \
    cudaError_t err = (call);                                      \
    if (err != cudaSuccess) {                                      \
        std::cerr << "CUDA error: " << cudaGetErrorString(err)     \
                  << " at " << __FILE__ << ":" << __LINE__         \
                  << std::endl;                                    \
        std::exit(1);                                              \
    }                                                              \
} while(0)

// CUDA-ядро: поэлементное сложение A и B -> C.
// Один поток обрабатывает один индекс idx.
__global__ void vec_add(const float* __restrict__ A,
                        const float* __restrict__ B,
                        float* __restrict__ C,
                        int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) C[idx] = A[idx] + B[idx];
}

// Функция замера среднего времени ядра для конкретного размера блока.
// Используем CUDA events + несколько повторов для стабильности.
float timeVecAdd(const float* dA, const float* dB, float* dC, int n,
                 int block, int iters)
{
    int grid = (n + block - 1) / block;

    // Прогрев: первый запуск + синхронизация
    vec_add<<<grid, block>>>(dA, dB, dC, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Замер времени серии запусков
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
    const int iters = 300;     // число повторов при замере времени

    // Набор размеров блока (минимум 3 по заданию)
    const int blocks[] = {128, 256, 512, 1024};
    const int numTests = sizeof(blocks) / sizeof(blocks[0]);

    // Массивы на CPU (A, B, C) и эталон (Ref)
    std::vector<float> hA(N), hB(N), hC(N), hRef(N);

    // Заполняем A и B случайными числами, Ref считаем на CPU
    std::mt19937 gen(123);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < N; ++i) {
        hA[i] = dist(gen);
        hB[i] = dist(gen);
        hRef[i] = hA[i] + hB[i];
    }

    // Память на GPU
    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
    CUDA_CHECK(cudaMalloc(&dA, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dB, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dC, N * sizeof(float)));

    // Копирование входных данных на GPU
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    // Заголовок таблицы результатов
    std::cout << "N = " << N << ", iters = " << iters << "\n\n";

    std::cout << std::left
              << std::setw(10) << "Block"
              << std::setw(10) << "Grid"
              << std::setw(20) << "Avg kernel (ms)"
              << "\n";
    std::cout << std::string(40, '-') << "\n";

    // Поиск лучшего размера блока (минимальное время)
    float bestMs = 1e9f;
    int bestBlock = -1;

    for (int t = 0; t < numTests; ++t) {
        int block = blocks[t];
        int grid  = (N + block - 1) / block;

        // Замер времени для текущего block
        float avgMs = timeVecAdd(dA, dB, dC, N, block, iters);

        // Печать строки таблицы
        std::cout << std::left
                  << std::setw(10) << block
                  << std::setw(10) << grid
                  << std::setw(20) << std::fixed << std::setprecision(6) << avgMs
                  << "\n";

        // Обновляем лучший результат
        if (avgMs < bestMs) {
            bestMs = avgMs;
            bestBlock = block;
        }
    }

    // Проверка результата: запуск с лучшим block и копирование на CPU
    int bestGrid = (N + bestBlock - 1) / bestBlock;
    vec_add<<<bestGrid, bestBlock>>>(dA, dB, dC, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(hC.data(), dC, N * sizeof(float), cudaMemcpyDeviceToHost));

    // Считаем максимальную абсолютную ошибку
    float maxErr = 0.0f;
    for (int i = 0; i < N; ++i) {
        float err = std::fabs(hC[i] - hRef[i]);
        if (err > maxErr) maxErr = err;
    }

    const float EPS = 1e-4f;
    bool ok = (maxErr <= EPS);

    // Итог: лучший размер блока + корректность
    std::cout << "\nBest block size = " << bestBlock
              << " (avg " << bestMs << " ms)\n";
    std::cout << "Max abs error = " << maxErr << "\n";
    std::cout << "Correct = " << (ok ? "YES" : "NO") << "\n";

    // Освобождение памяти GPU
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    return 0;
}
