#include <mpi.h>
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <numeric>

int main(int argc, char** argv)
{
    // Инициализация MPI
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // номер текущего процесса
    MPI_Comm_size(MPI_COMM_WORLD, &size); // общее количество процессов

    // Размер массива
    const int N = 1'000'000;

    std::vector<double> data;      // полный массив (только у rank 0)
    std::vector<int> counts(size); // сколько элементов отправляем каждому процессу
    std::vector<int> displs(size); // смещения для Scatterv

    // rank 0 создаёт данные 
    if (rank == 0)
    {
        data.resize(N);

        // Генератор случайных чисел
        std::mt19937 gen(42);
        std::uniform_real_distribution<double> dist(0.0, 1.0);

        for (int i = 0; i < N; ++i)
            data[i] = dist(gen);

        // Базовый размер части
        int base = N / size;
        int remainder = N % size;

        // Распределяем остаток
        for (int i = 0; i < size; ++i)
            counts[i] = base + (i < remainder ? 1 : 0);

        // Смещения
        displs[0] = 0;
        for (int i = 1; i < size; ++i)
            displs[i] = displs[i - 1] + counts[i - 1];
    }

    // Каждый процесс узнаёт, сколько элементов ему придёт
    int local_n = 0;
    MPI_Scatter(counts.data(), 1, MPI_INT,
                &local_n, 1, MPI_INT,
                0, MPI_COMM_WORLD);

    // Локальный буфер
    std::vector<double> local_data(local_n);

    //  Scatterv: распределяем массив 
    MPI_Scatterv(
        data.data(),        // отправляемый массив
        counts.data(),      // размеры частей
        displs.data(),      // смещения
        MPI_DOUBLE,
        local_data.data(),  // локальный массив
        local_n,
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD
    );

    // Локальные вычисления 
    double local_sum = 0.0;
    double local_sq_sum = 0.0;

    for (double x : local_data)
    {
        local_sum += x;
        local_sq_sum += x * x;
    }

    // Глобальные суммы 
    double global_sum = 0.0;
    double global_sq_sum = 0.0;

    MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_sq_sum, &global_sq_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    // Итоговые вычисления на rank 0 
    if (rank == 0)
    {
        double mean = global_sum / N;
        double variance = (global_sq_sum / N) - (mean * mean);
        double stddev = std::sqrt(variance);

        std::cout << "Mean value: " << mean << std::endl;
        std::cout << "Standard deviation: " << stddev << std::endl;
    }

    // Завершение MPI
    MPI_Finalize();
    return 0;
}
