#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>
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

// CPU reference (inclusive scan) 
static void cpu_inclusive_scan(const std::vector<float>& in, std::vector<float>& out)
{
    out.resize(in.size());
    double acc = 0.0;
    for (size_t i = 0; i < in.size(); ++i) {
        acc += (double)in[i];
        out[i] = (float)acc;
    }
}

// Device helpers: Blelloch exclusive scan in shared
// Делает EXCLUSIVE scan для массива длины blockDim.x (длина = power of 2 желательно).
// На выходе: s[tid] = sum(s[0..tid-1]) (exclusive)
__device__ void blelloch_exclusive_scan(float* s)
{
    int tid = threadIdx.x;
    int n = blockDim.x;

    // Up-sweep (reduce) phase
    for (int offset = 1; offset < n; offset <<= 1) {
        int idx = (tid + 1) * offset * 2 - 1;
        __syncthreads();
        if (idx < n) {
            s[idx] += s[idx - offset];
        }
    }

    // Set last element to 0
    __syncthreads();
    if (tid == 0) s[n - 1] = 0.0f;

    // Down-sweep phase
    for (int offset = n >> 1; offset > 0; offset >>= 1) {
        int idx = (tid + 1) * offset * 2 - 1;
        __syncthreads();
        if (idx < n) {
            float t = s[idx - offset];
            s[idx - offset] = s[idx];
            s[idx] += t;
        }
    }
    __syncthreads();
}

// 1) Скан внутри блока + запись суммы блока
__global__ void block_scan_kernel(const float* __restrict__ in,
                                  float* __restrict__ out,
                                  float* __restrict__ block_sums,
                                  int n)
{
    extern __shared__ float s[]; // shared memory

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    // Загружаем элемент или 0, если вышли за n
    float x = (gid < n) ? in[gid] : 0.0f;
    s[tid] = x;
    __syncthreads();

    // Делает exclusive scan в shared
    blelloch_exclusive_scan(s);

    // Переводим exclusive -> inclusive: inclusive = exclusive + original
    float inclusive = s[tid] + x;

    if (gid < n) out[gid] = inclusive;

    // Сумма блока = inclusive последнего реального элемента блока
    // Но если n не кратно блоку — надо аккуратно взять последний валидный.
    // Проще: сумма блока = сумма всех x в блоке (это exclusive_last + last_x), где "last_x" — x последнего потока в блоке (tid = blockDim-1), но может быть вне n.
    // Поэтому вычислим block_total как сумма всех x: это inclusive у tid=(blockDim-1) для полного блока, а для неполного — найдём последний валидный tid.
    __syncthreads();

    // Найдём сколько элементов реально в блоке:
    int block_start = blockIdx.x * blockDim.x;
    int valid = n - block_start;
    if (valid > blockDim.x) valid = blockDim.x;

    if (tid == 0) {
        float total = 0.0f;
        if (valid > 0) {
            // inclusive у последнего валидного элемента:
            int last_tid = valid - 1;
            total = (s[last_tid] + ((block_start + last_tid) < n ? in[block_start + last_tid] : 0.0f));
        }
        block_sums[blockIdx.x] = total;
    }
}

// 2) Добавить оффсет (prefix суммы блоков) ко всем элементам блока
__global__ void add_offsets_kernel(float* __restrict__ out,
                                   const float* __restrict__ block_offsets, // inclusive scan по block_sums
                                   int n)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;

    int b = blockIdx.x;
    float offset = (b == 0) ? 0.0f : block_offsets[b - 1]; // сумма всех предыдущих блоков
    out[gid] += offset;
}

// Рекурсивный scan для массива block_sums на GPU (делает inclusive scan)
static void gpu_inclusive_scan(float* d_in, float* d_out, int n, int threads)
{
    int blocks = (n + threads - 1) / threads;

    float* d_block_sums = nullptr;
    CUDA_CHECK(cudaMalloc(&d_block_sums, blocks * sizeof(float)));

    // 1) Скан внутри блоков для d_in -> d_out, и сохраняем суммы блоков
    block_scan_kernel<<<blocks, threads, threads * sizeof(float)>>>(d_in, d_out, d_block_sums, n);
    CUDA_CHECK(cudaGetLastError());

    if (blocks > 1) {
        // 2) Сканируем суммы блоков (рекурсивно)
        float* d_scanned_block_sums = nullptr;
        CUDA_CHECK(cudaMalloc(&d_scanned_block_sums, blocks * sizeof(float)));

        gpu_inclusive_scan(d_block_sums, d_scanned_block_sums, blocks, threads);

        // 3) Добавляем оффсеты к каждому блоку
        add_offsets_kernel<<<blocks, threads>>>(d_out, d_scanned_block_sums, n);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaFree(d_scanned_block_sums));
    }

    CUDA_CHECK(cudaFree(d_block_sums));
}

int main()
{
    // Тестовые размеры из задания можно брать 1024, 1e6, 1e7. :contentReference[oaicite:3]{index=3}
    const int N = 1'000'000;

    std::vector<float> h_in(N);
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < N; ++i) h_in[i] = dist(rng);

    // CPU reference
    std::vector<float> h_ref;
    cpu_inclusive_scan(h_in, h_ref);

    // GPU memory
    float *d_in = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    // threads должно быть степенью двойки для стабильной Blelloch-логики
    const int threads = 256;

    // Timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    // GPU scan
    gpu_inclusive_scan(d_in, d_out, N, threads);

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    // Copy back
    std::vector<float> h_out(N);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, N * sizeof(float), cudaMemcpyDeviceToHost));

    // Check correctness
    double max_abs_err = 0.0;
    for (int i = 0; i < N; ++i) {
        double e = std::fabs((double)h_out[i] - (double)h_ref[i]);
        if (e > max_abs_err) max_abs_err = e;
    }

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "N = " << N << "\n";
    std::cout << "Max abs error = " << max_abs_err << "\n";
    std::cout << "GPU scan time (ms) = " << ms << "\n";

    // Cleanup
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return 0;
}
