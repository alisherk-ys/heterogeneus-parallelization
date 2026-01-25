#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>
#include <cmath>

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) \
                  << " | " << __FILE__ << ":" << __LINE__ << "\n"; \
        std::exit(1); \
    } \
} while(0)

// CPU (последовательно): inclusive prefix sum 
void cpuInclusiveScan(const std::vector<float>& in, std::vector<float>& out)
{
    float s = 0.0f;
    for (size_t i = 0; i < in.size(); ++i) {
        s += in[i];
        out[i] = s;
    }
}

// GPU: scan одного блока в shared memory
// Мы берём 2 элемента на поток.
// 1 блок обрабатывает: 2 * blockDim.x элементов.
// Делаем EXCLUSIVE scan по Blelloch, затем превращаем в inclusive.
__global__ void blockScanInclusive(const float* __restrict__ d_in,
                                   float* __restrict__ d_out,
                                   float* __restrict__ d_blockSums,
                                   int n)
{
    extern __shared__ float sh[]; // размер: 2*blockDim.x

    const int t = threadIdx.x;
    const int base = 2 * blockIdx.x * blockDim.x;

    // Грузим 2 элемента в shared, если вылезли за n — кладём 0
    int i0 = base + t;
    int i1 = base + t + blockDim.x;

    float x0 = (i0 < n) ? d_in[i0] : 0.0f;
    float x1 = (i1 < n) ? d_in[i1] : 0.0f;

    sh[t] = x0;
    sh[t + blockDim.x] = x1;

    __syncthreads();

    // Blelloch scan (exclusive) 
    // Up-sweep (reduce)
    int offset = 1;
    int m = 2 * blockDim.x;
    for (int d = m >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (t < d) {
            int ai = offset * (2 * t + 1) - 1;
            int bi = offset * (2 * t + 2) - 1;
            sh[bi] += sh[ai];
        }
        offset <<= 1;
    }

    // В конце в sh[m-1] лежит сумма блока
    __syncthreads();
    float total = sh[m - 1];

    // Для exclusive scan последний элемент ставим в 0
    if (t == 0) sh[m - 1] = 0.0f;

    // Down-sweep
    for (int d = 1; d < m; d <<= 1) {
        offset >>= 1;
        __syncthreads();
        if (t < d) {
            int ai = offset * (2 * t + 1) - 1;
            int bi = offset * (2 * t + 2) - 1;
            float tmp = sh[ai];
            sh[ai] = sh[bi];
            sh[bi] += tmp;
        }
    }
    __syncthreads();

    // Сейчас sh[] = exclusive prefix sums.
    // Превратим в inclusive: inclusive[i] = exclusive[i] + input[i]
    if (i0 < n) d_out[i0] = sh[t] + x0;
    if (i1 < n) d_out[i1] = sh[t + blockDim.x] + x1;

    // Сохраним сумму блока (для смещений)
    if (d_blockSums && t == 0) {
        d_blockSums[blockIdx.x] = total;
    }
}

// Скан blockSums (их немного). Для 1e6 и block=512 -> blocks ~ 977? Нет,
// у нас 2*512=1024 элементов на блок, значит blocks ~ 977.
// blockSums длиной 977 помещается в один блок при threads=512 (2*512=1024).
__global__ void scanBlockSumsInclusive(const float* __restrict__ d_in,
                                      float* __restrict__ d_out,
                                      int nSums)
{
    extern __shared__ float sh[];

    int t = threadIdx.x;
    int m = 2 * blockDim.x;

    int i0 = t;
    int i1 = t + blockDim.x;

    float x0 = (i0 < nSums) ? d_in[i0] : 0.0f;
    float x1 = (i1 < nSums) ? d_in[i1] : 0.0f;

    sh[t] = x0;
    sh[t + blockDim.x] = x1;

    __syncthreads();

    // Blelloch exclusive
    int offset = 1;
    for (int d = m >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (t < d) {
            int ai = offset * (2 * t + 1) - 1;
            int bi = offset * (2 * t + 2) - 1;
            sh[bi] += sh[ai];
        }
        offset <<= 1;
    }

    __syncthreads();
    if (t == 0) sh[m - 1] = 0.0f;

    for (int d = 1; d < m; d <<= 1) {
        offset >>= 1;
        __syncthreads();
        if (t < d) {
            int ai = offset * (2 * t + 1) - 1;
            int bi = offset * (2 * t + 2) - 1;
            float tmp = sh[ai];
            sh[ai] = sh[bi];
            sh[bi] += tmp;
        }
    }
    __syncthreads();

    // inclusive для block sums
    if (i0 < nSums) d_out[i0] = sh[t] + x0;
    if (i1 < nSums) d_out[i1] = sh[t + blockDim.x] + x1;
}

// Добавляем смещения: для блока b нужно прибавить сумму всех предыдущих блоков.
// У нас d_scannedBlockSums — inclusive scan, поэтому offset для блока b:
// offset = (b == 0) ? 0 : d_scannedBlockSums[b-1]
__global__ void addBlockOffsets(float* d_data,
                               const float* __restrict__ d_scannedBlockSums,
                               int n)
{
    int b = blockIdx.x;
    float offset = (b == 0) ? 0.0f : d_scannedBlockSums[b - 1];

    int tid = threadIdx.x;
    int base = 2 * b * blockDim.x;

    int i0 = base + tid;
    int i1 = base + tid + blockDim.x;

    if (i0 < n) d_data[i0] += offset;
    if (i1 < n) d_data[i1] += offset;
}


int main()
{
    const int N = 1'000'000;

    // Данные
    std::vector<float> h_in(N), h_cpu(N), h_gpu(N);

    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < N; ++i) h_in[i] = dist(rng);

    // CPU timing 
    auto cpu_t0 = std::chrono::high_resolution_clock::now();
    cpuInclusiveScan(h_in, h_cpu);
    auto cpu_t1 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(cpu_t1 - cpu_t0).count();

    // GPU memory
    float *d_in = nullptr, *d_out = nullptr;
    float *d_blockSums = nullptr, *d_scannedBlockSums = nullptr;

    CUDA_CHECK(cudaMalloc(&d_in,  N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    // Настройки: 512 потоков => 1024 элемента на блок (2 на поток)
    const int threads = 512;
    const int elemsPerBlock = 2 * threads;
    int blocks = (N + elemsPerBlock - 1) / elemsPerBlock;

    CUDA_CHECK(cudaMalloc(&d_blockSums, blocks * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_scannedBlockSums, blocks * sizeof(float)));

    // GPU timing (ядра) 
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    // 1) scan по блокам + block sums
    blockScanInclusive<<<blocks, threads, elemsPerBlock * sizeof(float)>>>(
        d_in, d_out, d_blockSums, N
    );
    CUDA_CHECK(cudaGetLastError());

    // 2) scan block sums (один блок, потому что blocks <= 1024 здесь)
    //    Для N=1e6 и threads=512: blocks примерно 977, влезает.
    int sumsThreads = 512;
    int sumsShared = 2 * sumsThreads * (int)sizeof(float);

    scanBlockSumsInclusive<<<1, sumsThreads, sumsShared>>>(
        d_blockSums, d_scannedBlockSums, blocks
    );
    CUDA_CHECK(cudaGetLastError());

    // 3) добавить offset каждому блоку
    addBlockOffsets<<<blocks, threads>>>(d_out, d_scannedBlockSums, N);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float gpu_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, start, stop));

    // Copy back + check 
    CUDA_CHECK(cudaMemcpy(h_gpu.data(), d_out, N * sizeof(float), cudaMemcpyDeviceToHost));

    // Проверка точности (достаточно посмотреть max abs error)
    float max_abs_err = 0.0f;
    for (int i = 0; i < N; ++i) {
        float e = std::fabs(h_cpu[i] - h_gpu[i]);
        if (e > max_abs_err) max_abs_err = e;
    }

    // Вывод результатов 
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "N = " << N << "\n";
    std::cout << "Blocks = " << blocks << ", Threads = " << threads
              << " (elements per block = " << elemsPerBlock << ")\n\n";

    std::cout << "CPU time (ms): " << cpu_ms << "\n";
    std::cout << "GPU time (ms): " << gpu_ms << "  (scan + scan block sums + add offsets)\n";
    std::cout << "Max abs error: " << max_abs_err << "\n";

    // Освобождаем ресурсы
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_blockSums));
    CUDA_CHECK(cudaFree(d_scannedBlockSums));

    return 0;
}

