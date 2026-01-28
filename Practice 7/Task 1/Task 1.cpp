#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <numeric>
#include <iomanip>
#include <cmath>

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = (call);                                            \
        if (err != cudaSuccess) {                                            \
            std::cerr << "CUDA error: " << cudaGetErrorString(err)           \
                      << " at " << __FILE__ << ":" << __LINE__ << "\n";      \
            std::exit(1);                                                    \
        }                                                                    \
    } while (0)

// Ядро редукции: каждый блок считает частичную сумму.
// - читаем данные из глобальной памяти
// - складываем в shared memory
// - внутри блока делаем редукцию (пополам) до одного значения
__global__ void reduce_sum_kernel(const float* __restrict__ input,
                                  float* __restrict__ block_sums,
                                  int n)
{
    extern __shared__ float sdata[]; // shared memory для текущего блока

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    // Каждый поток грузит до 2 элементов (grid-stride по 2*blockDim)
    float sum = 0.0f;
    if (i < (unsigned)n) sum += input[i];
    if (i + blockDim.x < (unsigned)n) sum += input[i + blockDim.x];

    // Кладём в shared
    sdata[tid] = sum;
    __syncthreads();

    // Редукция внутри блока (binary tree reduction)
    // blockDim.x должно быть степенью двойки (мы так и зададим)
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Поток 0 записывает сумму блока
    if (tid == 0) {
        block_sums[blockIdx.x] = sdata[0];
    }
}

// CPU-версия суммы (для проверки)
static double cpu_sum(const std::vector<float>& x)
{
    // double - чтобы на CPU было точнее
    double s = 0.0;
    for (float v : x) s += (double)v;
    return s;
}

int main()
{
    // 1) Тестовый массив
    // Можно поставить 1024 (маленький тест), 1'000'000 и т.д.
    const int N = 1'000'000;

    std::vector<float> h_x(N);

    // Генерация случайных чисел
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < N; ++i) h_x[i] = dist(rng);

    // CPU контроль
    double cpu = cpu_sum(h_x);

    // 2) Выделяем память на GPU 
    float* d_in = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, h_x.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    // Параметры запуска
    // threads = 256 — типичная настройка; обязательно степень двойки
    int threads = 256;

    // В первом проходе каждый блок обрабатывает 2*threads элементов
    int blocks = (N + (threads * 2 - 1)) / (threads * 2);

    float* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out, blocks * sizeof(float)));

    // Таймер GPU
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    // === 3) Многошаговая редукция ===
    // Сначала считаем суммы блоков из исходного массива
    int cur_n = N;
    const float* cur_in = d_in;
    float* cur_out = d_out;

    while (true) {
        int cur_blocks = (cur_n + (threads * 2 - 1)) / (threads * 2);

        // shared memory = threads * sizeof(float)
        reduce_sum_kernel<<<cur_blocks, threads, threads * sizeof(float)>>>(cur_in, cur_out, cur_n);
        CUDA_CHECK(cudaGetLastError());

        // Если остался 1 блок — это финальная сумма в cur_out[0]
        if (cur_blocks == 1) break;

        // Иначе: теперь входом станет массив частичных сумм
        cur_n = cur_blocks;

        // Переиспользуем буферы:
        // Чтобы не выделять каждый раз, можно "пинг-понг" через два буфера.
        // Здесь проще: выделим новый буфер под следующий выход.
        float* next_out = nullptr;
        int next_blocks = (cur_n + (threads * 2 - 1)) / (threads * 2);
        CUDA_CHECK(cudaMalloc(&next_out, next_blocks * sizeof(float)));

        // Освобождаем старый вход, если он был не исходным d_in
        if (cur_in != d_in) {
            CUDA_CHECK(cudaFree((void*)cur_in));
        }

        cur_in = cur_out;
        cur_out = next_out;
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    // === 4) Копируем результат на CPU ===
    float gpu_result_f = 0.0f;
    CUDA_CHECK(cudaMemcpy(&gpu_result_f, cur_out, sizeof(float), cudaMemcpyDeviceToHost));
    double gpu = (double)gpu_result_f;

    // === 5) Проверка корректности ===
    double abs_err = std::fabs(cpu - gpu);
    double rel_err = abs_err / (std::fabs(cpu) + 1e-12);

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "N = " << N << "\n";
    std::cout << "CPU sum = " << cpu << "\n";
    std::cout << "GPU sum = " << gpu << "\n";
    std::cout << "Abs error = " << abs_err << "\n";
    std::cout << "Rel error = " << rel_err << "\n";
    std::cout << "GPU reduction time (ms) = " << ms << "\n";

    // === 6) Очистка ===
    if (cur_in != d_in) CUDA_CHECK(cudaFree((void*)cur_in));
    CUDA_CHECK(cudaFree(cur_out));
    CUDA_CHECK(cudaFree(d_in));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return 0;
}

