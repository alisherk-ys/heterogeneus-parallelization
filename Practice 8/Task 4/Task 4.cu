#include <cuda_runtime.h>
#include <omp.h>

#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <iomanip>
#include <algorithm>
#include <cmath>
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
// Умножаем каждый элемент на 2
__global__ void mul2_kernel(float* data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] *= 2.0f;
}

// Умножаем на 2 во второй половине (offset..offset+n-1)
__global__ void mul2_kernel_offset(float* data, int offset, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[offset + i] *= 2.0f;
}

// Вспомогательные функции 
static void fill_data(std::vector<float>& a, int seed = 123) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (auto& x : a) x = dist(gen);
}

static bool check_mul2(const std::vector<float>& before, const std::vector<float>& after, float eps = 1e-5f) {
    if (before.size() != after.size()) return false;
    for (size_t i = 0; i < before.size(); ++i) {
        float expected = before[i] * 2.0f;
        if (std::fabs(after[i] - expected) > eps * (1.0f + std::fabs(expected))) return false;
    }
    return true;
}

// CPU (OpenMP) 
static double run_cpu_openmp(std::vector<float>& a) {
    auto t0 = std::chrono::high_resolution_clock::now();

    #pragma omp parallel for
    for (int i = 0; i < (int)a.size(); ++i) {
        a[i] *= 2.0f;
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

// GPU (CUDA end-to-end) 
// Возвращает время: H2D + kernel + D2H (в мс)
static double run_gpu_end_to_end(const std::vector<float>& in, std::vector<float>& out) {
    int N = (int)in.size();
    size_t bytes = (size_t)N * sizeof(float);

    float* d = nullptr;
    CUDA_CHECK(cudaMalloc(&d, bytes));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    int block = 256;
    int grid  = (N + block - 1) / block;

    // Время "от и до"
    CUDA_CHECK(cudaEventRecord(start));

    // H2D
    CUDA_CHECK(cudaMemcpy(d, in.data(), bytes, cudaMemcpyHostToDevice));

    // kernel
    mul2_kernel<<<grid, block>>>(d, N);
    CUDA_CHECK(cudaGetLastError());

    // D2H
    out.resize(N);
    CUDA_CHECK(cudaMemcpy(out.data(), d, bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d));

    return (double)ms;
}

// Hybrid (CPU+GPU одновременно, end-to-end) 
// Первая половина: CPU OpenMP
// Вторая половина: GPU (H2D второй половины + kernel + D2H второй половины)
static double run_hybrid_end_to_end(std::vector<float>& a) {
    int N = (int)a.size();
    int half = N / 2;
    size_t half_bytes = (size_t)(N - half) * sizeof(float); // вторая часть (если N нечетное)

    float* d_second = nullptr;
    CUDA_CHECK(cudaMalloc(&d_second, half_bytes));

    int second_n = N - half;
    int block = 256;
    int grid  = (second_n + block - 1) / block;

    auto t0 = std::chrono::high_resolution_clock::now();

    // Параллельно: CPU работает над [0..half-1], GPU над [half..N-1]
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            // CPU: первая половина
            #pragma omp parallel for
            for (int i = 0; i < half; ++i) {
                a[i] *= 2.0f;
            }
        }

        #pragma omp section
        {
            // GPU: вторая половина (end-to-end только для второй половины)
            CUDA_CHECK(cudaMemcpy(d_second, a.data() + half, half_bytes, cudaMemcpyHostToDevice));
            mul2_kernel<<<grid, block>>>(d_second, second_n);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(a.data() + half, d_second, half_bytes, cudaMemcpyDeviceToHost));
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();

    CUDA_CHECK(cudaFree(d_second));

    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

int main() {
    // Инфо о GPU
    int dev = 0;
    CUDA_CHECK(cudaSetDevice(dev));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    std::cout << "GPU: " << prop.name << "\n";

    // Размеры для теста (можно расширить)
    std::vector<int> sizes = {
        1 << 16,   // 65536
        1 << 18,   // 262144
        1 << 20,   // 1048576
        1 << 22    // 4194304
    };

    // Чтобы замеры были стабильнее: прогрев + несколько повторов
    const int repeats = 5;

    for (int N : sizes) {
        std::vector<float> base(N);
        fill_data(base, 123);

        // CPU 
        double cpu_ms_best = 1e100;
        std::vector<float> cpu_out;

        for (int r = 0; r < repeats; ++r) {
            cpu_out = base;
            double ms = run_cpu_openmp(cpu_out);
            cpu_ms_best = std::min(cpu_ms_best, ms);
        }

        // GPU 
        double gpu_ms_best = 1e100;
        std::vector<float> gpu_out;

        // прогрев (один раз)
        {
            std::vector<float> warm_out;
            (void)run_gpu_end_to_end(base, warm_out);
        }

        for (int r = 0; r < repeats; ++r) {
            double ms = run_gpu_end_to_end(base, gpu_out);
            gpu_ms_best = std::min(gpu_ms_best, ms);
        }

        // Hybrid 
        double hyb_ms_best = 1e100;
        std::vector<float> hyb_out;

        for (int r = 0; r < repeats; ++r) {
            hyb_out = base;
            double ms = run_hybrid_end_to_end(hyb_out);
            hyb_ms_best = std::min(hyb_ms_best, ms);
        }

        // Проверка корректности
        bool ok_cpu = check_mul2(base, cpu_out);
        bool ok_gpu = check_mul2(base, gpu_out);
        bool ok_hyb = check_mul2(base, hyb_out);

        // Ускорения относительно CPU
        double speed_gpu = cpu_ms_best / gpu_ms_best;
        double speed_hyb = cpu_ms_best / hyb_ms_best;

        // Вывод списком
        std::cout << "\nArray size: " << N << "\n";
        std::cout << "  CPU (OpenMP) time:      " << std::fixed << std::setprecision(3) << cpu_ms_best << " ms"
                  << "   | check: " << (ok_cpu ? "OK" : "FAIL") << "\n";
        std::cout << "  GPU (CUDA) time:        " << gpu_ms_best << " ms"
                  << "   | check: " << (ok_gpu ? "OK" : "FAIL") << "\n";
        std::cout << "  Hybrid (CPU+GPU) time:  " << hyb_ms_best << " ms"
                  << "   | check: " << (ok_hyb ? "OK" : "FAIL") << "\n";

        std::cout << "  Speedup vs CPU:\n";
        std::cout << "    GPU speedup:          " << speed_gpu << "x\n";
        std::cout << "    Hybrid speedup:       " << speed_hyb << "x\n";

        // Кто лучший
        double best = std::min({cpu_ms_best, gpu_ms_best, hyb_ms_best});
        std::string winner = (best == cpu_ms_best) ? "CPU" : (best == gpu_ms_best ? "GPU" : "Hybrid");
        std::cout << "  Best mode: " << winner << "\n";
        std::cout << std::string(60, '-') << "\n";
    }
    return 0;
}
