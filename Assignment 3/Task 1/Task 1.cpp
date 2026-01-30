#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include <cmath>

// Макрос для проверки ошибок CUDA.
// При ошибке выводит сообщение и завершает программу.
#define CUDA_CHECK(call) do {                                      \
    cudaError_t err = (call);                                      \
    if (err != cudaSuccess) {                                      \
        std::cerr << "CUDA error: " << cudaGetErrorString(err)     \
                  << " at " << __FILE__ << ":" << __LINE__         \
                  << std::endl;                                    \
        std::exit(1);                                              \
    }                                                              \
} while(0)

// ================= CUDA-ЯДРА =================

// 1) Ядро с использованием только глобальной памяти.
// Каждый поток умножает один элемент массива на коэффициент k.
__global__ void mul_global(const float* __restrict__ in,
                           float* __restrict__ out,
                           float k,
                           int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        out[idx] = in[idx] * k;
}

// 2) Ядро с использованием разделяемой (shared) памяти.
// Данные блока сначала загружаются в shared память,
// затем выполняется умножение и запись результата.
__global__ void mul_shared(const float* __restrict__ in,
                           float* __restrict__ out,
                           float k,
                           int n)
{
    extern __shared__ float s[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    if (idx < n)
        s[tid] = in[idx];

    __syncthreads();

    if (idx < n)
        s[tid] *= k;

    __syncthreads();

    if (idx < n)
        out[idx] = s[tid];
}

// ============ ФУНКЦИЯ ЗАМЕРА ВРЕМЕНИ ============

// Измеряет среднее время выполнения CUDA-ядра
// с использованием CUDA events.
template <typename LaunchFunc>
float timeKernel(LaunchFunc launch, int iters) {

    // Прогрев GPU
    launch();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) {
        launch();
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return ms / iters;
}

int main() {
    const int N = 1'000'000;   // Размер массива
    const float k = 3.5f;     // Коэффициент умножения

    // Массивы на стороне CPU
    std::vector<float> h_in(N), h_out(N), h_ref(N);

    // Генерация входных данных
    std::mt19937 gen(123);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < N; ++i)
        h_in[i] = dist(gen);

    // Эталонный результат на CPU
    for (int i = 0; i < N; ++i)
        h_ref[i] = h_in[i] * k;

    // Выделение памяти на GPU
    float *d_in = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in,  N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(float)));

    // Копирование данных на GPU
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(),
                           N * sizeof(float),
                           cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemset(d_out, 0, N * sizeof(float)));

    // Конфигурация CUDA
    const int block = 256;
    const int grid  = (N + block - 1) / block;
    const int iters = 200;
    const size_t shmem = block * sizeof(float);

    // Замер времени ядра с глобальной памятью
    float global_ms = timeKernel([&](){
        mul_global<<<grid, block>>>(d_in, d_out, k, N);
    }, iters);

    // Замер времени ядра с shared памятью
    float shared_ms = timeKernel([&](){
        mul_shared<<<grid, block, shmem>>>(d_in, d_out, k, N);
    }, iters);

    // Копирование результата обратно на CPU
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out,
                           N * sizeof(float),
                           cudaMemcpyDeviceToHost));

    // Проверка корректности результата
    float max_err = 0.0f;
    for (int i = 0; i < N; ++i)
        max_err = std::max(max_err, std::fabs(h_out[i] - h_ref[i]));

    const float EPS = 1e-4f;

    // Вывод результатов
    std::cout << "Global kernel time (ms): " << global_ms << "\n";
    std::cout << "Shared kernel time (ms): " << shared_ms << "\n";
    std::cout << "Max abs error: " << max_err << "\n";
    std::cout << "Correct: " << (max_err <= EPS ? "YES" : "NO") << "\n";
    std::cout << "Speed ratio (shared/global): "
              << (shared_ms / global_ms) << "\n";

    // Освобождение памяти
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));

    return 0;
}
