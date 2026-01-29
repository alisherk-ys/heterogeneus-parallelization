#define CL_TARGET_OPENCL_VERSION 120

#include <CL/cl.h>

#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <chrono>
#include <random>
#include <cmath>

// Утилита: безопасный вывод ошибок OpenCL
static const char* cl_errstr(cl_int err) {
    switch (err) {
        case CL_SUCCESS: return "CL_SUCCESS";
        case CL_DEVICE_NOT_FOUND: return "CL_DEVICE_NOT_FOUND";
        case CL_DEVICE_NOT_AVAILABLE: return "CL_DEVICE_NOT_AVAILABLE";
        case CL_COMPILER_NOT_AVAILABLE: return "CL_COMPILER_NOT_AVAILABLE";
        case CL_MEM_OBJECT_ALLOCATION_FAILURE: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
        case CL_OUT_OF_RESOURCES: return "CL_OUT_OF_RESOURCES";
        case CL_OUT_OF_HOST_MEMORY: return "CL_OUT_OF_HOST_MEMORY";
        case CL_PROFILING_INFO_NOT_AVAILABLE: return "CL_PROFILING_INFO_NOT_AVAILABLE";
        case CL_MEM_COPY_OVERLAP: return "CL_MEM_COPY_OVERLAP";
        case CL_IMAGE_FORMAT_MISMATCH: return "CL_IMAGE_FORMAT_MISMATCH";
        case CL_IMAGE_FORMAT_NOT_SUPPORTED: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
        case CL_BUILD_PROGRAM_FAILURE: return "CL_BUILD_PROGRAM_FAILURE";
        case CL_MAP_FAILURE: return "CL_MAP_FAILURE";
        case CL_INVALID_VALUE: return "CL_INVALID_VALUE";
        case CL_INVALID_DEVICE_TYPE: return "CL_INVALID_DEVICE_TYPE";
        case CL_INVALID_PLATFORM: return "CL_INVALID_PLATFORM";
        case CL_INVALID_DEVICE: return "CL_INVALID_DEVICE";
        case CL_INVALID_CONTEXT: return "CL_INVALID_CONTEXT";
        case CL_INVALID_QUEUE_PROPERTIES: return "CL_INVALID_QUEUE_PROPERTIES";
        case CL_INVALID_COMMAND_QUEUE: return "CL_INVALID_COMMAND_QUEUE";
        case CL_INVALID_HOST_PTR: return "CL_INVALID_HOST_PTR";
        case CL_INVALID_MEM_OBJECT: return "CL_INVALID_MEM_OBJECT";
        case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
        case CL_INVALID_IMAGE_SIZE: return "CL_INVALID_IMAGE_SIZE";
        case CL_INVALID_SAMPLER: return "CL_INVALID_SAMPLER";
        case CL_INVALID_BINARY: return "CL_INVALID_BINARY";
        case CL_INVALID_BUILD_OPTIONS: return "CL_INVALID_BUILD_OPTIONS";
        case CL_INVALID_PROGRAM: return "CL_INVALID_PROGRAM";
        case CL_INVALID_PROGRAM_EXECUTABLE: return "CL_INVALID_PROGRAM_EXECUTABLE";
        case CL_INVALID_KERNEL_NAME: return "CL_INVALID_KERNEL_NAME";
        case CL_INVALID_KERNEL_DEFINITION: return "CL_INVALID_KERNEL_DEFINITION";
        case CL_INVALID_KERNEL: return "CL_INVALID_KERNEL";
        case CL_INVALID_ARG_INDEX: return "CL_INVALID_ARG_INDEX";
        case CL_INVALID_ARG_VALUE: return "CL_INVALID_ARG_VALUE";
        case CL_INVALID_ARG_SIZE: return "CL_INVALID_ARG_SIZE";
        case CL_INVALID_KERNEL_ARGS: return "CL_INVALID_KERNEL_ARGS";
        case CL_INVALID_WORK_DIMENSION: return "CL_INVALID_WORK_DIMENSION";
        case CL_INVALID_WORK_GROUP_SIZE: return "CL_INVALID_WORK_GROUP_SIZE";
        case CL_INVALID_WORK_ITEM_SIZE: return "CL_INVALID_WORK_ITEM_SIZE";
        case CL_INVALID_GLOBAL_OFFSET: return "CL_INVALID_GLOBAL_OFFSET";
        case CL_INVALID_EVENT_WAIT_LIST: return "CL_INVALID_EVENT_WAIT_LIST";
        case CL_INVALID_EVENT: return "CL_INVALID_EVENT";
        case CL_INVALID_OPERATION: return "CL_INVALID_OPERATION";
        case CL_INVALID_GL_OBJECT: return "CL_INVALID_GL_OBJECT";
        case CL_INVALID_BUFFER_SIZE: return "CL_INVALID_BUFFER_SIZE";
        case CL_INVALID_MIP_LEVEL: return "CL_INVALID_MIP_LEVEL";
        default: return "Unknown OpenCL error";
    }
}

// Читаем весь файл в строку
static std::string read_text_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open file: " + path);
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

static void die_if(cl_int err, const char* where) {
    if (err != CL_SUCCESS) {
        std::cerr << "[OpenCL ERROR] " << where << " -> " << cl_errstr(err) << " (" << err << ")\n";
        std::exit(1);
    }
}

int main(int argc, char** argv) {
    // 0) Выбор типа устройства (cpu/gpu)
    cl_device_type desired = CL_DEVICE_TYPE_GPU;
    if (argc >= 2) {
        std::string mode = argv[1];
        if (mode == "cpu") desired = CL_DEVICE_TYPE_CPU;
        if (mode == "gpu") desired = CL_DEVICE_TYPE_GPU;
    }

    // Размер данных (можешь увеличить для теста производительности)
    const int n = 1 << 22; // ~4 млн float
    const size_t bytes = sizeof(float) * (size_t)n;

    // 1) Подготовка данных A и B
    std::vector<float> A(n), B(n), C(n), C_ref(n);

    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int i = 0; i < n; ++i) {
        A[i] = dist(rng);
        B[i] = dist(rng);
    }

    // CPU-референс для проверки корректности (не обязательно, но полезно для отчёта)
    for (int i = 0; i < n; ++i) C_ref[i] = A[i] + B[i];


    // 2) Поиск платформы и устройства OpenCL
    cl_int err = CL_SUCCESS;

    cl_uint num_platforms = 0;
    die_if(clGetPlatformIDs(0, nullptr, &num_platforms), "clGetPlatformIDs(count)");
    if (num_platforms == 0) {
        std::cerr << "OpenCL platforms not found. Проверь драйверы OpenCL.\n";
        return 1;
    }

    std::vector<cl_platform_id> platforms(num_platforms);
    die_if(clGetPlatformIDs(num_platforms, platforms.data(), nullptr), "clGetPlatformIDs(list)");

    cl_platform_id chosen_platform = nullptr;
    cl_device_id chosen_device = nullptr;

    // Пробуем найти устройство нужного типа на любой платформе
    for (auto p : platforms) {
        cl_uint num_devices = 0;
        err = clGetDeviceIDs(p, desired, 0, nullptr, &num_devices);
        if (err != CL_SUCCESS || num_devices == 0) continue;

        std::vector<cl_device_id> devices(num_devices);
        die_if(clGetDeviceIDs(p, desired, num_devices, devices.data(), nullptr), "clGetDeviceIDs(list)");
        chosen_platform = p;
        chosen_device = devices[0];
        break;
    }

    // Если не нашли (например, GPU нет) — попробуем CPU как запасной вариант
    if (!chosen_device) {
        std::cerr << "Не найдено устройство типа " << (desired == CL_DEVICE_TYPE_GPU ? "GPU" : "CPU")
                  << ". Пробую альтернативу...\n";
        cl_device_type fallback = (desired == CL_DEVICE_TYPE_GPU) ? CL_DEVICE_TYPE_CPU : CL_DEVICE_TYPE_GPU;

        for (auto p : platforms) {
            cl_uint num_devices = 0;
            err = clGetDeviceIDs(p, fallback, 0, nullptr, &num_devices);
            if (err != CL_SUCCESS || num_devices == 0) continue;

            std::vector<cl_device_id> devices(num_devices);
            die_if(clGetDeviceIDs(p, fallback, num_devices, devices.data(), nullptr), "clGetDeviceIDs(fallback)");
            chosen_platform = p;
            chosen_device = devices[0];
            desired = fallback;
            break;
        }
    }

    if (!chosen_device) {
        std::cerr << "Не найдено ни CPU, ни GPU OpenCL устройства.\n";
        return 1;
    }

    // Выведем имя устройства (удобно для отчёта)
    {
        char name[256]{};
        die_if(clGetDeviceInfo(chosen_device, CL_DEVICE_NAME, sizeof(name), name, nullptr), "clGetDeviceInfo(NAME)");
        std::cout << "Selected device: " << name
                  << " (" << (desired == CL_DEVICE_TYPE_GPU ? "GPU" : "CPU") << ")\n";
    }


    // 3) Контекст и командная очередь
    cl_context_properties props[] = {
        CL_CONTEXT_PLATFORM, (cl_context_properties)chosen_platform,
        0
    };

    cl_context ctx = clCreateContext(props, 1, &chosen_device, nullptr, nullptr, &err);
    die_if(err, "clCreateContext");

    // Очередь с профилированием, чтобы замерять время kernel (GPU/CPU)
    cl_command_queue queue = clCreateCommandQueue(ctx, chosen_device, CL_QUEUE_PROFILING_ENABLE, &err);
    die_if(err, "clCreateCommandQueue");


    // 4) Загружаем и компилируем ядро (kernel.cl)
    std::string src = read_text_file("kernel.cl");
    const char* src_ptr = src.c_str();
    size_t src_len = src.size();

    cl_program program = clCreateProgramWithSource(ctx, 1, &src_ptr, &src_len, &err);
    die_if(err, "clCreateProgramWithSource");

    // Компилируем
    err = clBuildProgram(program, 1, &chosen_device, "", nullptr, nullptr);
    if (err != CL_SUCCESS) {
        // Если сборка упала - покажем лог
        size_t log_size = 0;
        clGetProgramBuildInfo(program, chosen_device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        std::string log(log_size, '\0');
        clGetProgramBuildInfo(program, chosen_device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
        std::cerr << "Build log:\n" << log << "\n";
        die_if(err, "clBuildProgram");
    }

    cl_kernel kernel = clCreateKernel(program, "vector_add", &err);
    die_if(err, "clCreateKernel(vector_add)");

  
    // 5) Буферы в памяти устройства (GPU/CPU OpenCL runtime)
    cl_mem bufA = clCreateBuffer(ctx, CL_MEM_READ_ONLY,  bytes, nullptr, &err); die_if(err, "clCreateBuffer(A)");
    cl_mem bufB = clCreateBuffer(ctx, CL_MEM_READ_ONLY,  bytes, nullptr, &err); die_if(err, "clCreateBuffer(B)");
    cl_mem bufC = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, bytes, nullptr, &err); die_if(err, "clCreateBuffer(C)");

    // Копируем A и B на устройство
    die_if(clEnqueueWriteBuffer(queue, bufA, CL_TRUE, 0, bytes, A.data(), 0, nullptr, nullptr), "WriteBuffer(A)");
    die_if(clEnqueueWriteBuffer(queue, bufB, CL_TRUE, 0, bytes, B.data(), 0, nullptr, nullptr), "WriteBuffer(B)");


    // 6) Аргументы ядра
    die_if(clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufA), "clSetKernelArg(0)");
    die_if(clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufB), "clSetKernelArg(1)");
    die_if(clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufC), "clSetKernelArg(2)");
    die_if(clSetKernelArg(kernel, 3, sizeof(int), &n),       "clSetKernelArg(3)");


    // 7) Запуск ядра + замер времени kernel через профилирование событий
    size_t global = (size_t)n;
    // local можно оставить nullptr (runtime сам подберёт) — проще для практики
    // но для отчёта можно поэкспериментировать: например local = 256
    cl_event evt{};
    auto t0 = std::chrono::high_resolution_clock::now();

    die_if(clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global, nullptr, 0, nullptr, &evt),
           "clEnqueueNDRangeKernel");

    die_if(clFinish(queue), "clFinish");

    auto t1 = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // Время именно kernel из профилирования (если поддерживается)
    cl_ulong start_ns = 0, end_ns = 0;
    double kernel_ms = -1.0;
    if (clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_START, sizeof(start_ns), &start_ns, nullptr) == CL_SUCCESS &&
        clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_END,   sizeof(end_ns),   &end_ns,   nullptr) == CL_SUCCESS) {
        kernel_ms = (double)(end_ns - start_ns) * 1e-6; // ns -> ms
    }


    // 8) Чтение результата обратно на host
    die_if(clEnqueueReadBuffer(queue, bufC, CL_TRUE, 0, bytes, C.data(), 0, nullptr, nullptr), "ReadBuffer(C)");


    // 9) Проверка корректности
    double max_abs_err = 0.0;
    for (int i = 0; i < n; ++i) {
        double e = std::fabs((double)C[i] - (double)C_ref[i]);
        if (e > max_abs_err) max_abs_err = e;
    }

    std::cout << "Max abs error: " << max_abs_err << "\n";
    std::cout << "Total time (enqueue+finish) ms: " << total_ms << "\n";
    if (kernel_ms >= 0.0) {
        std::cout << "Kernel time (event profiling) ms: " << kernel_ms << "\n";
    } else {
        std::cout << "Kernel time: profiling not available\n";
    }


    // 10) Освобождение ресурсов OpenCL
    clReleaseEvent(evt);
    clReleaseMemObject(bufA);
    clReleaseMemObject(bufB);
    clReleaseMemObject(bufC);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);

    std::cout << "Done.\n";
    return 0;
}
