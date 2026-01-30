#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <omp.h>
#include <iomanip>
#include <cstdlib>

#define CUDA_CHECK(call) do {                                  \
    cudaError_t err = (call);                                  \
    if (err != cudaSuccess) {                                  \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) \
                  << " at " << __FILE__ << ":" << __LINE__     \
                  << std::endl;                                \
        std::exit(1);                                          \
    }                                                          \
} while(0)

// CUDA kernel
// Обрабатывает вторую половину массива: a[i] *= 2
__global__ void gpu_process(float* data, int offset, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = offset + i;

    if (i < n) {
        data[idx] *= 2.0f;
    }
}

int main() {
    // 1. Исходные данные 
    const int N = 1'000'000;   // Размер массива
    const int half = N / 2;    // Половина массива
    const size_t bytes = N * sizeof(float);

    std::vector<float> h_data(N);

    // Инициализация массива
    for (int i = 0; i < N; ++i) {
        h_data[i] = static_cast<float>(i);
    }

    // 2. Выделение памяти на GPU 
    float* d_data = nullptr;
    CUDA_CHECK(cudaMalloc(&d_data, bytes));

    // Копируем ВЕСЬ массив на GPU (проще для примера)
    CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), bytes, cudaMemcpyHostToDevice));

    // 3. Настройка CUDA 
    const int blockSize = 256;
    const int gridSize  = (half + blockSize - 1) / blockSize;

    // 4. Замер общего времени 
    auto start = std::chrono::high_resolution_clock::now();

    // 5. CPU и GPU работают одновременно 
    #pragma omp parallel sections
    {
        // CPU часть 
        #pragma omp section
        {
            // Обрабатываем первую половину массива
            for (int i = 0; i < half; ++i) {
                h_data[i] *= 2.0f;
            }
        }

        // GPU часть 
        #pragma omp section
        {
            // Запуск kernel для второй половины массива
            gpu_process<<<gridSize, blockSize>>>(d_data, half, half);
            CUDA_CHECK(cudaDeviceSynchronize()); // Ждём GPU
        }
    }

    // 6. Копирование результата GPU обратно 
    CUDA_CHECK(cudaMemcpy(h_data.data() + half,
                          d_data + half,
                          half * sizeof(float),
                          cudaMemcpyDeviceToHost));

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> total_ms = end - start;

    // 7. Проверка корректности 
    bool ok = true;
    for (int i = 0; i < N; ++i) {
        if (h_data[i] != 2.0f * i) {
            ok = false;
            break;
        }
    }

    // 8. Вывод результатов 
    std::cout << "Array size: " << N << "\n";
    std::cout << "Hybrid processing time (CPU + GPU): "
              << std::fixed << std::setprecision(3)
              << total_ms.count() << " ms\n";
    std::cout << "Result check: " << (ok ? "OK" : "FAILED") << "\n";

    // 9. Освобождение памяти 
    CUDA_CHECK(cudaFree(d_data));

    return 0;
}
