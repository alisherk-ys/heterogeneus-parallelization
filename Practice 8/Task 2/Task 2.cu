#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <iomanip>

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
// Каждый поток обрабатывает один элемент массива: a[i] *= 2
__global__ void multiply_by_two(float* a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        a[i] *= 2.0f;
    }
}

int main() {
    // 1) Создаем данные на CPU 
    const int N = 1'000'000;                 // Размер массива
    const size_t bytes = N * sizeof(float);  // Сколько байт нужно

    std::vector<float> h_data(N);

    // Инициализация массива: h_data[i] = i
    for (int i = 0; i < N; ++i) {
        h_data[i] = static_cast<float>(i);
    }

    // 2) Выделяем память на GPU 
    float* d_data = nullptr;
    CUDA_CHECK(cudaMalloc(&d_data, bytes));

    // 3) Копируем массив на GPU 
    CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), bytes, cudaMemcpyHostToDevice));

    // 4) Настройка запуска kernel 
    const int blockSize = 256;                         // Потоков в блоке
    const int gridSize  = (N + blockSize - 1) / blockSize;  // Кол-во блоков

    // 5) Замер времени kernel на GPU 
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    // Запуск ядра
    multiply_by_two<<<gridSize, blockSize>>>(d_data, N);

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop)); // ждём завершения kernel

    // Проверяем ошибки kernel
    CUDA_CHECK(cudaGetLastError());

    // Получаем время в миллисекундах
    float kernel_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, start, stop));

    // 6) Копируем результат обратно на CPU 
    CUDA_CHECK(cudaMemcpy(h_data.data(), d_data, bytes, cudaMemcpyDeviceToHost));

    // 7) Вывод результата и простая проверка 
    std::cout << "Array size: " << N << "\n";
    std::cout << "GPU kernel time: " << std::fixed << std::setprecision(4)
              << kernel_ms << " ms\n";

    // Проверим несколько значений
    std::cout << "Check values:\n";
    std::cout << "  h_data[0]     = " << h_data[0] << " (expected 0)\n";
    std::cout << "  h_data[1]     = " << h_data[1] << " (expected 2)\n";
    std::cout << "  h_data[N-1]   = " << h_data[N-1]
              << " (expected " << 2.0f * (N - 1) << ")\n";

    // 8) Освобождение памяти 
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_data));

    return 0;
}
