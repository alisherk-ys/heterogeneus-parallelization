#include <mpi.h>
#include <iostream>
#include <vector>
#include <random>
#include <numeric>
#include <iomanip>

// Локальная обработка массива (пример вычисления)
// Можно заменить на любую свою "обработку", главное — чтобы была одинаковая логика на всех процессах.
static void process_chunk(const float* x, float* y, int n, float a, float b)
{
    for (int i = 0; i < n; ++i) {
        y[i] = a * x[i] + b;
    }
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Размер массива можно передать аргументом: mpirun -np 4 ./mpi_prog 10000000
    long long N = 10'000'000; // по умолчанию 10 млн (чтобы MPI реально имел смысл)
    if (argc >= 2) {
        N = std::stoll(argv[1]);
        if (N < 1) N = 1;
    }

    const float A = 1.2345f;
    const float B = -0.9876f;

    // готовим разбиение N между процессами
    // counts[i] = сколько элементов получит i-й процесс
    // displs[i] = с какого индекса в глобальном массиве начинается кусок i-го процесса
    std::vector<int> counts(size, 0);
    std::vector<int> displs(size, 0);

    long long base = N / size;
    long long rem  = N % size;

    for (int i = 0; i < size; ++i) {
        long long cnt = base + (i < rem ? 1 : 0);
        counts[i] = static_cast<int>(cnt); // для очень огромных N нужно MPIX/long long, но для учебы обычно ок
    }
    displs[0] = 0;
    for (int i = 1; i < size; ++i) {
        displs[i] = displs[i - 1] + counts[i - 1];
    }

    // глобальные массивы только на rank 0 
    std::vector<float> x_global;
    std::vector<float> y_global;

    if (rank == 0) {
        x_global.resize(N);
        y_global.resize(N);

        // Заполняем входные данные
        std::mt19937 rng(123);
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        for (long long i = 0; i < N; ++i) {
            x_global[i] = dist(rng);
        }
    }

    // локальные буферы 
    int local_n = counts[rank];
    std::vector<float> x_local(local_n);
    std::vector<float> y_local(local_n);

    // Важно: честный замер часто делают так:
    // 1) Barrier (чтобы все стартовали вместе)
    // 2) t0
    // 3) Scatterv + compute + gatherv
    // 4) t1
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    // Раздать куски x
    MPI_Scatterv(
        rank == 0 ? x_global.data() : nullptr,
        counts.data(),
        displs.data(),
        MPI_FLOAT,
        x_local.data(),
        local_n,
        MPI_FLOAT,
        0,
        MPI_COMM_WORLD
    );

    // Локальная обработка
    process_chunk(x_local.data(), y_local.data(), local_n, A, B);

    // Собрать y обратно
    MPI_Gatherv(
        y_local.data(),
        local_n,
        MPI_FLOAT,
        rank == 0 ? y_global.data() : nullptr,
        counts.data(),
        displs.data(),
        MPI_FLOAT,
        0,
        MPI_COMM_WORLD
    );

    double t1 = MPI_Wtime();
    double local_time = t1 - t0;

    // Возьмём максимальное время среди процессов (обычно это и есть время параллельного алгоритма)
    double total_time = 0.0;
    MPI_Reduce(&local_time, &total_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Дополнительно: контрольная сумма результата (чтобы проверить, что все процессы реально работали)
    double local_checksum = 0.0;
    for (int i = 0; i < local_n; ++i) local_checksum += y_local[i];

    double global_checksum = 0.0;
    MPI_Reduce(&local_checksum, &global_checksum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "MPI processes: " << size << "\n";
        std::cout << "N: " << N << "\n";
        std::cout << "Time (max over ranks), sec: " << total_time << "\n";
        std::cout << "Checksum (sum of y), approx: " << global_checksum << "\n";
        std::cout << "Chunk sizes example: ";
        for (int i = 0; i < std::min(size, 8); ++i) std::cout << counts[i] << (i + 1 < std::min(size, 8) ? ", " : "");
        if (size > 8) std::cout << ", ...";
        std::cout << "\n";
    }

    MPI_Finalize();
    return 0;
}
