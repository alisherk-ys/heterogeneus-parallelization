#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <algorithm>
#include <iomanip>
#include <cassert>
#include <cmath>

#define CUDA_CHECK(call) do {                                      \
    cudaError_t err = (call);                                      \
    if (err != cudaSuccess) {                                      \
        std::cerr << "CUDA error: " << cudaGetErrorString(err)     \
                  << " at " << __FILE__ << ":" << __LINE__         \
                  << std::endl;                                    \
        std::exit(1);                                              \
    }                                                              \
} while(0)

// CPU (эталон)

// CPU редукция (сумма)
static float cpu_reduce_sum(const std::vector<float>& a) {
    double s = 0.0;
    for (float x : a) s += x;
    return (float)s;
}

// CPU inclusive scan: out[i] = a[0] + ... + a[i]
static void cpu_inclusive_scan(const std::vector<float>& a, std::vector<float>& out) {
    out.resize(a.size());
    double run = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        run += a[i];
        out[i] = (float)run;
    }
}

// GPU REDUCTION 
// Наивный вариант (плохой):
// - потоки читают global память со stride
// - затем atomicAdd в global (дорого и конфликтно)
__global__ void reduce_naive_global_atomic(const float* in, float* partial, int n) {
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    int stride = blockDim.x * gridDim.x;

    float local = 0.0f;
    for (int i = gid; i < n; i += stride) {
        local += in[i];
    }

    // Обнулим partial[blockIdx.x]
    if (tid == 0) partial[blockIdx.x] = 0.0f;
    __syncthreads();

    // Очень дорого: atomic на global памяти
    atomicAdd(&partial[blockIdx.x], local);
}

// Оптимизированная редукция: shared memory внутри блока
__global__ void reduce_shared_optimized(const float* __restrict__ in,
                                       float* __restrict__ partial,
                                       int n)
{
    extern __shared__ float sdata[]; // blockDim.x элементов

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    // Коалесцированное чтение
    float x = (gid < n) ? in[gid] : 0.0f;
    sdata[tid] = x;
    __syncthreads();

    // Редукция деревом в shared
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0) partial[blockIdx.x] = sdata[0];
}

// Многораундовая редукция: пока не останется 1 элемент.
// kernel_ms — сумма времени всех kernel запусков.
static float gpu_reduce_sum(const float* d_in, int n, bool optimized, float& kernel_ms) {
    kernel_ms = 0.0f;

    const int block = 256;
    int grid = std::min((n + block - 1) / block, 1024);

    float* d_partial = nullptr;
    CUDA_CHECK(cudaMalloc(&d_partial, grid * sizeof(float)));

    const float* d_cur_in = d_in;
    int cur_n = n;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    while (true) {
        grid = std::min((cur_n + block - 1) / block, 1024);

        CUDA_CHECK(cudaEventRecord(start));

        if (!optimized) {
            reduce_naive_global_atomic<<<grid, block>>>(d_cur_in, d_partial, cur_n);
        } else {
            reduce_shared_optimized<<<grid, block, block * sizeof(float)>>>(d_cur_in, d_partial, cur_n);
        }

        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        kernel_ms += ms;

        CUDA_CHECK(cudaGetLastError());

        if (grid == 1) break;

        // Следующий раунд: вход = partial, размер = grid
        d_cur_in = d_partial;
        cur_n = grid;

        // Нужен новый выходной буфер, чтобы не затирать вход
        int next_grid = std::min((cur_n + block - 1) / block, 1024);
        float* d_next = nullptr;
        CUDA_CHECK(cudaMalloc(&d_next, next_grid * sizeof(float)));

        std::swap(d_partial, d_next);
        CUDA_CHECK(cudaFree(d_next));
    }

    float result = 0.0f;
    CUDA_CHECK(cudaMemcpy(&result, d_partial, sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_partial));

    return result;
}

// GPU SCAN
// Inclusive scan (prefix sum)
// A) Наивный Hillis–Steele (global): много проходов, ping-pong буферы.
__global__ void scan_hillis_steele_step(const float* in, float* out, int n, int offset) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float val = in[i];
        if (i >= offset) val += in[i - offset];
        out[i] = val;
    }
}

static void gpu_scan_naive_global(const float* d_in, float* d_out, int n, float& kernel_ms) {
    kernel_ms = 0.0f;

    float* d_tmp = nullptr;
    CUDA_CHECK(cudaMalloc(&d_tmp, n * sizeof(float)));

    // out = in
    CUDA_CHECK(cudaMemcpy(d_out, d_in, n * sizeof(float), cudaMemcpyDeviceToDevice));

    int block = 256;
    int grid = (n + block - 1) / block;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    bool ping = true;
    for (int offset = 1; offset < n; offset <<= 1) {
        CUDA_CHECK(cudaEventRecord(start));

        if (ping) {
            scan_hillis_steele_step<<<grid, block>>>(d_out, d_tmp, n, offset);
        } else {
            scan_hillis_steele_step<<<grid, block>>>(d_tmp, d_out, n, offset);
        }

        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        kernel_ms += ms;

        CUDA_CHECK(cudaGetLastError());
        ping = !ping;
    }

    // Если результат остался в tmp - копируем обратно
    if (!ping) {
        CUDA_CHECK(cudaMemcpy(d_out, d_tmp, n * sizeof(float), cudaMemcpyDeviceToDevice));
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_tmp));
}

// B) Оптимизированный scan: Blelloch в shared по блокам + оффсеты блоков.
// Kernel 1: внутри блока (shared) делаем inclusive scan и сохраняем сумму блока.
__global__ void scan_blelloch_block_inclusive(const float* __restrict__ in,
                                              float* __restrict__ out,
                                              float* __restrict__ block_sums,
                                              int n)
{
    extern __shared__ float temp[]; // blockDim.x элементов

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    float x = (gid < n) ? in[gid] : 0.0f;
    temp[tid] = x;
    __syncthreads();

    // Up-sweep
    for (int offset = 1; offset < blockDim.x; offset <<= 1) {
        int idx = (tid + 1) * offset * 2 - 1;
        if (idx < blockDim.x) {
            temp[idx] += temp[idx - offset];
        }
        __syncthreads();
    }

    // Сохраняем сумму блока, делаем exclusive основу
    if (tid == blockDim.x - 1) {
        block_sums[blockIdx.x] = temp[tid];
        temp[tid] = 0.0f;
    }
    __syncthreads();

    // Down-sweep
    for (int offset = blockDim.x >> 1; offset >= 1; offset >>= 1) {
        int idx = (tid + 1) * offset * 2 - 1;
        if (idx < blockDim.x) {
            float t = temp[idx - offset];
            temp[idx - offset] = temp[idx];
            temp[idx] += t;
        }
        __syncthreads();
    }

    // temp - exclusive, делаем inclusive: + x
    if (gid < n) out[gid] = temp[tid] + x;
}

// Kernel 2: добавляем оффсет блока ко всем его элементам
__global__ void add_block_offsets(float* data, const float* block_offsets, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {
        data[gid] += block_offsets[blockIdx.x];
    }
}

static void gpu_scan_optimized_shared(const float* d_in, float* d_out, int n, float& kernel_ms) {
    kernel_ms = 0.0f;

    const int block = 256;
    int numBlocks = (n + block - 1) / block;

    float* d_block_sums = nullptr;
    CUDA_CHECK(cudaMalloc(&d_block_sums, numBlocks * sizeof(float)));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // 1) scan по блокам + суммы блоков
    CUDA_CHECK(cudaEventRecord(start));
    scan_blelloch_block_inclusive<<<numBlocks, block, block * sizeof(float)>>>(d_in, d_out, d_block_sums, n);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms1 = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms1, start, stop));
    kernel_ms += ms1;

    CUDA_CHECK(cudaGetLastError());

    // 2) считаем оффсеты блоков на CPU (массив маленький)
    std::vector<float> h_block_sums(numBlocks);
    CUDA_CHECK(cudaMemcpy(h_block_sums.data(), d_block_sums, numBlocks * sizeof(float), cudaMemcpyDeviceToHost));

    std::vector<float> h_offsets(numBlocks, 0.0f);
    double run = 0.0;
    for (int b = 0; b < numBlocks; ++b) {
        h_offsets[b] = (float)run; // сумма всех предыдущих блоков
        run += h_block_sums[b];
    }

    float* d_offsets = nullptr;
    CUDA_CHECK(cudaMalloc(&d_offsets, numBlocks * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_offsets, h_offsets.data(), numBlocks * sizeof(float), cudaMemcpyHostToDevice));

    // 3) добавляем оффсеты блоков
    CUDA_CHECK(cudaEventRecord(start));
    add_block_offsets<<<numBlocks, block>>>(d_out, d_offsets, n);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms2 = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms2, start, stop));
    kernel_ms += ms2;

    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_offsets));
    CUDA_CHECK(cudaFree(d_block_sums));
}

// Утилиты 

static void fill_random(std::vector<float>& a, int seed = 123) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (auto& x : a) x = dist(gen);
}

static bool almost_equal(float a, float b, float eps = 1e-2f) {
    return std::abs(a - b) <= eps * (1.0f + std::max(std::abs(a), std::abs(b)));
}

static float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    assert(a.size() == b.size());
    float m = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        m = std::max(m, std::abs(a[i] - b[i]));
    }
    return m;
}

// main 

int main() {
    // Выбираем GPU
    int dev = 0;
    CUDA_CHECK(cudaSetDevice(dev));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    std::cout << "GPU: " << prop.name << "\n\n";

    // Размеры массивов для эксперимента
    std::vector<int> sizes = {
        1 << 16,  // 65536
        1 << 18,  // 262144
        1 << 20,  // 1048576
        1 << 22   // 4194304
    };

    for (int n : sizes) {
        // Данные на CPU 
        std::vector<float> h_in(n);
        fill_random(h_in, 123);

        // CPU: редукция 
        float cpu_sum = 0.0f;
        double cpu_red_ms = 0.0;
        {
            auto t0 = std::chrono::high_resolution_clock::now();
            cpu_sum = cpu_reduce_sum(h_in);
            auto t1 = std::chrono::high_resolution_clock::now();
            cpu_red_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        }

        // CPU: scan 
        std::vector<float> h_cpu_scan;
        double cpu_sc_ms = 0.0;
        {
            auto t0 = std::chrono::high_resolution_clock::now();
            cpu_inclusive_scan(h_in, h_cpu_scan);
            auto t1 = std::chrono::high_resolution_clock::now();
            cpu_sc_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        }

        // Копирование на GPU 
        float* d_in = nullptr;
        float* d_out = nullptr;
        CUDA_CHECK(cudaMalloc(&d_in, n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), n * sizeof(float), cudaMemcpyHostToDevice));

        // GPU: редукция (naive vs shared)
        float gpu_red_na_ms = 0.0f, gpu_red_sh_ms = 0.0f;
        float gpu_sum_na = gpu_reduce_sum(d_in, n, /*optimized=*/false, gpu_red_na_ms);
        float gpu_sum_sh = gpu_reduce_sum(d_in, n, /*optimized=*/true,  gpu_red_sh_ms);

        // Проверка корректности редукции
        if (!almost_equal(cpu_sum, gpu_sum_na) || !almost_equal(cpu_sum, gpu_sum_sh)) {
            std::cerr << "Reduce mismatch at N=" << n
                      << " cpu=" << cpu_sum
                      << " gpu_na=" << gpu_sum_na
                      << " gpu_sh=" << gpu_sum_sh << "\n";
        }

        // GPU: scan (naive vs shared) 
        float gpu_sc_na_ms = 0.0f, gpu_sc_sh_ms = 0.0f;

        // Наивный scan
        gpu_scan_naive_global(d_in, d_out, n, gpu_sc_na_ms);
        std::vector<float> h_gpu_scan_na(n);
        CUDA_CHECK(cudaMemcpy(h_gpu_scan_na.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost));
        float diff_na = max_abs_diff(h_cpu_scan, h_gpu_scan_na);

        // Оптимизированный scan
        gpu_scan_optimized_shared(d_in, d_out, n, gpu_sc_sh_ms);
        std::vector<float> h_gpu_scan_sh(n);
        CUDA_CHECK(cudaMemcpy(h_gpu_scan_sh.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost));
        float diff_sh = max_abs_diff(h_cpu_scan, h_gpu_scan_sh);

        if (diff_na > 1e-2f || diff_sh > 1e-2f) {
            std::cerr << "Scan mismatch at N=" << n
                      << " diff_na=" << diff_na
                      << " diff_sh=" << diff_sh << "\n";
                      std::cout << "\n";
        }

        // Вывод результатов 
        std::cout << "Array size: " << n << "\n";

        std::cout << "  Reduction (sum):\n";
        std::cout << "    CPU:                     "
                  << std::fixed << std::setprecision(3) << cpu_red_ms << " ms\n";
        std::cout << "    GPU (naive, global):      "
                  << gpu_red_na_ms << " ms\n";
        std::cout << "    GPU (optimized, shared):  "
                  << gpu_red_sh_ms << " ms\n";

        std::cout << "  Scan (prefix sum):\n";
        std::cout << "    CPU:                     "
                  << cpu_sc_ms << " ms\n";
        std::cout << "    GPU (naive, global):      "
                  << gpu_sc_na_ms << " ms\n";
        std::cout << "    GPU (optimized, shared):  "
                  << gpu_sc_sh_ms << " ms\n";

        std::cout << std::string(60, '-') << "\n";

        // Освобождение памяти 
        CUDA_CHECK(cudaFree(d_in));
        CUDA_CHECK(cudaFree(d_out));
    }
    return 0;
}
