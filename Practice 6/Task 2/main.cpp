#define CL_TARGET_OPENCL_VERSION 120

#include <CL/cl.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <random>
#include <chrono>
#include <cmath>
#include <iomanip>

// Макрос для проверки ошибок OpenCL
#define CL_CHECK(err, msg) \
    if ((err) != CL_SUCCESS) { \
        std::cerr << "OpenCL error (" << (err) << ") at " << msg << std::endl; \
        std::exit(1); \
    }

// Загрузка текста kernel.cl из файла
static std::string load_text_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::cerr << "Cannot open file: " << path << std::endl;
        std::exit(1);
    }
    return std::string((std::istreambuf_iterator<char>(f)),
                        std::istreambuf_iterator<char>());
}

// Получение имени платформы OpenCL
static std::string get_platform_name(cl_platform_id pid) {
    size_t size = 0;
    clGetPlatformInfo(pid, CL_PLATFORM_NAME, 0, nullptr, &size);
    std::string name(size, '\0');
    clGetPlatformInfo(pid, CL_PLATFORM_NAME, size, name.data(), nullptr);
    return name;
}

// Получение имени устройства OpenCL
static std::string get_device_name(cl_device_id did) {
    size_t size = 0;
    clGetDeviceInfo(did, CL_DEVICE_NAME, 0, nullptr, &size);
    std::string name(size, '\0');
    clGetDeviceInfo(did, CL_DEVICE_NAME, size, name.data(), nullptr);
    return name;
}

// Поиск устройства нужного типа (GPU или CPU)
static bool pick_device(cl_device_type type,
                        cl_platform_id& out_platform,
                        cl_device_id& out_device)
{
    cl_uint platform_count = 0;
    clGetPlatformIDs(0, nullptr, &platform_count);

    if (platform_count == 0) return false;

    std::vector<cl_platform_id> platforms(platform_count);
    clGetPlatformIDs(platform_count, platforms.data(), nullptr);

    for (auto p : platforms) {
        cl_uint device_count = 0;
        if (clGetDeviceIDs(p, type, 0, nullptr, &device_count) != CL_SUCCESS)
            continue;

        if (device_count > 0) {
            std::vector<cl_device_id> devices(device_count);
            clGetDeviceIDs(p, type, device_count, devices.data(), nullptr);
            out_platform = p;
            out_device = devices[0];
            return true;
        }
    }
    return false;
}

// Последовательное матричное умножение на CPU
void cpu_matmul(const std::vector<float>& A,
                const std::vector<float>& B,
                std::vector<float>& C,
                int N, int M, int K)
{
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < K; ++j) {
            float sum = 0.0f;
            for (int t = 0; t < M; ++t) {
                sum += A[i * M + t] * B[t * K + j];
            }
            C[i * K + j] = sum;
        }
    }
}

int main() {
    std::cout << "OpenCL Matrix Multiplication\n";

    // Размеры матриц
    const int N = 512;
    const int M = 512;
    const int K = 512;

    // Попытка выбрать GPU, иначе CPU
    cl_platform_id platform = nullptr;
    cl_device_id device = nullptr;

    bool has_gpu = pick_device(CL_DEVICE_TYPE_GPU, platform, device);
    if (!has_gpu) {
        pick_device(CL_DEVICE_TYPE_CPU, platform, device);
    }

    std::cout << "Platform: " << get_platform_name(platform) << "\n";
    std::cout << "Device:   " << get_device_name(device) << "\n";

    cl_int err;

    // Создание контекста OpenCL
    cl_context_properties props[] = {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)platform,
        0
    };
    cl_context context = clCreateContext(props, 1, &device, nullptr, nullptr, &err);
    CL_CHECK(err, "clCreateContext");

    // Создание очереди команд с профилированием
    cl_command_queue queue =
        clCreateCommandQueue(context, device,
                             CL_QUEUE_PROFILING_ENABLE, &err);
    CL_CHECK(err, "clCreateCommandQueue");

    // Загрузка и компиляция ядра
    std::string source = load_text_file("kernel_matmul.cl");
    const char* src = source.c_str();
    size_t src_size = source.size();

    cl_program program =
        clCreateProgramWithSource(context, 1, &src, &src_size, &err);
    CL_CHECK(err, "clCreateProgramWithSource");

    CL_CHECK(clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr),
             "clBuildProgram");

    // Создание ядра
    cl_kernel kernel = clCreateKernel(program, "matmul_basic", &err);
    CL_CHECK(err, "clCreateKernel");

    // Подготовка данных
    std::vector<float> A(N * M);
    std::vector<float> B(M * K);
    std::vector<float> C(N * K, 0.0f);
    std::vector<float> Cref(N * K, 0.0f);

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (auto& x : A) x = dist(gen);
    for (auto& x : B) x = dist(gen);

    // Создание буферов OpenCL
    cl_mem bufA = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                 sizeof(float) * A.size(), nullptr, &err);
    cl_mem bufB = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                 sizeof(float) * B.size(), nullptr, &err);
    cl_mem bufC = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                 sizeof(float) * C.size(), nullptr, &err);

    // Копирование данных на устройство
    clEnqueueWriteBuffer(queue, bufA, CL_TRUE, 0,
                          sizeof(float) * A.size(), A.data(),
                          0, nullptr, nullptr);
    clEnqueueWriteBuffer(queue, bufB, CL_TRUE, 0,
                          sizeof(float) * B.size(), B.data(),
                          0, nullptr, nullptr);

    // Передача аргументов в ядро
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufA);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufB);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufC);
    clSetKernelArg(kernel, 3, sizeof(int), &N);
    clSetKernelArg(kernel, 4, sizeof(int), &M);
    clSetKernelArg(kernel, 5, sizeof(int), &K);

    // Размеры сетки
    size_t local[2]  = {16, 16};
    size_t global[2] = {
        ((size_t)N + local[0] - 1) / local[0] * local[0],
        ((size_t)K + local[1] - 1) / local[1] * local[1]
    };

    // Запуск ядра
    cl_event event;
    clEnqueueNDRangeKernel(queue, kernel, 2, nullptr,
                           global, local, 0, nullptr, &event);
    clWaitForEvents(1, &event);

    // Считывание результата
    clEnqueueReadBuffer(queue, bufC, CL_TRUE, 0,
                         sizeof(float) * C.size(), C.data(),
                         0, nullptr, nullptr);

    // Последовательная CPU-проверка
    auto t0 = std::chrono::high_resolution_clock::now();
    cpu_matmul(A, B, Cref, N, M, K);
    auto t1 = std::chrono::high_resolution_clock::now();

    double cpu_time =
        std::chrono::duration<double, std::milli>(t1 - t0).count();

    // Проверка ошибки
    float max_err = 0.0f;
    for (size_t i = 0; i < C.size(); ++i) {
        max_err = std::max(max_err, std::fabs(C[i] - Cref[i]));
    }

    std::cout << "CPU time: " << cpu_time << " ms\n";
    std::cout << "Max abs error: " << max_err << "\n";

    // Освобождение ресурсов
    clReleaseMemObject(bufA);
    clReleaseMemObject(bufB);
    clReleaseMemObject(bufC);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    return 0;
}
