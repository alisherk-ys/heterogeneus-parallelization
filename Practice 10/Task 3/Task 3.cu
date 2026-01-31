#include <cuda_runtime.h>   
#include <iostream>         
#include <iomanip>          
#include <thread>           
#include <cmath>            
#include <algorithm>        

// Макрос проверки ошибок CUDA
#define CUDA_CHECK(call) do {                                  \
  cudaError_t e = (call);                                      \
  if (e != cudaSuccess) {                                      \
    std::cerr << "CUDA error: " << cudaGetErrorString(e)       \
              << " at " << __FILE__ << ":" << __LINE__ << "\n";\
    std::exit(1);                                              \
  }                                                            \
} while(0)

// GPU-ядро: простая обработка массива
__global__ void process_kernel(const float* __restrict__ in,
                               float* __restrict__ out,
                               float k, int n)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) out[idx] = in[idx] * k + 1.0f;
}

// CPU-обработка части массива
static void cpu_process(const float* in, float* out, float k, int start, int end) {
  for (int i = start; i < end; ++i)
    out[i] = in[i] * k + 1.0f;
}

// Максимальная абсолютная ошибка
static float max_abs_err(const float* a, const float* b, int n) {
  float m = 0.0f;
  for (int i = 0; i < n; ++i)
    m = std::max(m, std::fabs(a[i] - b[i]));
  return m;
}

int main(int argc, char** argv) {
  // Параметры задачи
  const int N = (argc >= 2) ? std::max(1, std::atoi(argv[1])) : 10'000'000;
  const float k = (argc >= 3) ? (float)std::atof(argv[2]) : 3.5f;

  // Делим работу между CPU и GPU
  const int split = N / 2;
  const int N_gpu = N - split;

  std::cout << "Op: out[i] = in[i]*k + 1\n";
  std::cout << "N=" << N << " k=" << k << " split=" << split
            << " (CPU " << split << ", GPU " << N_gpu << ")\n\n";

  // Pinned host memory для асинхронных копий
  float *h_in=nullptr, *h_out=nullptr, *h_ref=nullptr;
  CUDA_CHECK(cudaHostAlloc(&h_in,  N * sizeof(float), cudaHostAllocDefault));
  CUDA_CHECK(cudaHostAlloc(&h_out, N * sizeof(float), cudaHostAllocDefault));
  CUDA_CHECK(cudaHostAlloc(&h_ref, N * sizeof(float), cudaHostAllocDefault));

  // Инициализация данных на CPU
  for (int i = 0; i < N; ++i) {
    h_in[i]  = (float)((i % 1000) / 1000.0);
    h_ref[i] = h_in[i] * k + 1.0f;
  }

  // Device-память только под GPU-часть
  float *d_in=nullptr, *d_out=nullptr;
  CUDA_CHECK(cudaMalloc(&d_in,  N_gpu * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out, N_gpu * sizeof(float)));

  // CUDA stream
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  // CUDA events для профилирования
  cudaEvent_t e_h2d_s, e_h2d_e, e_ker_s, e_ker_e, e_d2h_s, e_d2h_e;
  CUDA_CHECK(cudaEventCreate(&e_h2d_s));
  CUDA_CHECK(cudaEventCreate(&e_h2d_e));
  CUDA_CHECK(cudaEventCreate(&e_ker_s));
  CUDA_CHECK(cudaEventCreate(&e_ker_e));
  CUDA_CHECK(cudaEventCreate(&e_d2h_s));
  CUDA_CHECK(cudaEventCreate(&e_d2h_e));

  // Общее время выполнения
  auto t_total0 = std::chrono::high_resolution_clock::now();

  // 1) CPU-часть в отдельном потоке
  auto t_cpu0 = std::chrono::high_resolution_clock::now();
  std::thread cpu_thr([&](){
    cpu_process(h_in, h_out, k, 0, split);
  });

  // 2) Асинхронная копия Host → Device
  CUDA_CHECK(cudaEventRecord(e_h2d_s, stream));
  CUDA_CHECK(cudaMemcpyAsync(d_in, h_in + split, N_gpu * sizeof(float),
                             cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaEventRecord(e_h2d_e, stream));

  // 3) Запуск CUDA-ядра
  const int block = 256;
  const int grid = (N_gpu + block - 1) / block;

  CUDA_CHECK(cudaEventRecord(e_ker_s, stream));
  process_kernel<<<grid, block, 0, stream>>>(d_in, d_out, k, N_gpu);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(e_ker_e, stream));

  // 4) Асинхронная копия Device → Host
  CUDA_CHECK(cudaEventRecord(e_d2h_s, stream));
  CUDA_CHECK(cudaMemcpyAsync(h_out + split, d_out, N_gpu * sizeof(float),
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaEventRecord(e_d2h_e, stream));

  // 5) Ожидание GPU
  CUDA_CHECK(cudaStreamSynchronize(stream));

  // 6) Ожидание CPU
  cpu_thr.join();
  auto t_cpu1 = std::chrono::high_resolution_clock::now();

  auto t_total1 = std::chrono::high_resolution_clock::now();

  // Подсчёт времен
  float h2d_ms=0, ker_ms=0, d2h_ms=0;
  CUDA_CHECK(cudaEventElapsedTime(&h2d_ms, e_h2d_s, e_h2d_e));
  CUDA_CHECK(cudaEventElapsedTime(&ker_ms, e_ker_s, e_ker_e));
  CUDA_CHECK(cudaEventElapsedTime(&d2h_ms, e_d2h_s, e_d2h_e));

  double cpu_ms = std::chrono::duration<double, std::milli>(t_cpu1 - t_cpu0).count();
  double total_ms = std::chrono::duration<double, std::milli>(t_total1 - t_total0).count();

  float transfer_ms = h2d_ms + d2h_ms;
  double overlap_est = total_ms - std::max(cpu_ms, (double)(h2d_ms + ker_ms + d2h_ms));

  // Проверка корректности
  float err = max_abs_err(h_out, h_ref, N);

  // Вывод результатов
  std::cout << std::fixed << std::setprecision(6);
  std::cout << "CPU part time (ms)              = " << cpu_ms << "\n";
  std::cout << "GPU H2D time (ms)               = " << h2d_ms << "\n";
  std::cout << "GPU kernel time (ms)            = " << ker_ms << "\n";
  std::cout << "GPU D2H time (ms)               = " << d2h_ms << "\n";
  std::cout << "Transfer overhead (H2D+D2H) (ms) = " << transfer_ms << "\n";
  std::cout << "GPU pipeline (H2D+K+D2H) (ms)    = " << (h2d_ms + ker_ms + d2h_ms) << "\n";
  std::cout << "Total hybrid time (ms)           = " << total_ms << "\n";
  std::cout << "Overlap estimate (ms)            = " << overlap_est << "\n";
  std::cout << "Max abs error                    = " << err << "\n";
  std::cout << "Correct                          = " << (err < 1e-6f ? "Yes" : "No") << "\n";

  // Очистка ресурсов
  CUDA_CHECK(cudaEventDestroy(e_h2d_s));
  CUDA_CHECK(cudaEventDestroy(e_h2d_e));
  CUDA_CHECK(cudaEventDestroy(e_ker_s));
  CUDA_CHECK(cudaEventDestroy(e_ker_e));
  CUDA_CHECK(cudaEventDestroy(e_d2h_s));
  CUDA_CHECK(cudaEventDestroy(e_d2h_e));
  CUDA_CHECK(cudaStreamDestroy(stream));
  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));
  CUDA_CHECK(cudaFreeHost(h_in));
  CUDA_CHECK(cudaFreeHost(h_out));
  CUDA_CHECK(cudaFreeHost(h_ref));
  return 0;
}
