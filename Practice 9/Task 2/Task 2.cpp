#include <mpi.h>
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <iomanip>
#include <algorithm>

// Удобная печать матрицы (для маленьких N)
static void print_augmented(const std::vector<double>& Ab, int N) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            std::cout << std::setw(10) << std::setprecision(4) << Ab[i*(N+1) + j] << " ";
        }
        std::cout << " | " << std::setw(10) << std::setprecision(4) << Ab[i*(N+1) + N] << "\n";
    }
}

// Создаём "хорошую" систему: диагонально-доминантную матрицу, чтобы не ловить нулевые pivots.
static void make_diagonally_dominant_system(std::vector<double>& Ab, int N, unsigned seed = 42) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    // Ab хранит расширенную матрицу [A|b] размером N x (N+1)
    Ab.assign(N * (N + 1), 0.0);

    for (int i = 0; i < N; ++i) {
        double row_sum = 0.0;
        for (int j = 0; j < N; ++j) {
            double v = dist(gen);
            Ab[i*(N+1) + j] = v;
            row_sum += std::abs(v);
        }
        // Усиливаем диагональ: |a_ii| > sum(|a_ij|) (примерно)
        Ab[i*(N+1) + i] += row_sum + 1.0;

        // Правая часть b
        Ab[i*(N+1) + N] = dist(gen);
    }
}

// По распределению строк (counts/displs) возвращаем rank-владельца глобальной строки gRow
static int owner_of_row(int gRow, const std::vector<int>& counts, const std::vector<int>& displs) {
    // displs[p]..displs[p]+counts[p)-1 — диапазон строк процесса p
    for (int p = 0; p < (int)counts.size(); ++p) {
        if (gRow >= displs[p] && gRow < displs[p] + counts[p]) return p;
    }
    return 0; // на всякий случай
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Чтение N из аргументов 
    int N = 256;
    if (argc >= 2) N = std::max(2, std::atoi(argv[1]));

    // Подготовка распределения строк 
    // counts[p] = сколько строк у процесса p
    // displs[p] = с какого глобального индекса строки начинается блок процесса p
    std::vector<int> counts(size), displs(size);
    int base = N / size;
    int rem  = N % size;

    for (int p = 0; p < size; ++p) counts[p] = base + (p < rem ? 1 : 0);
    displs[0] = 0;
    for (int p = 1; p < size; ++p) displs[p] = displs[p - 1] + counts[p - 1];

    // rank 0 создаёт систему 
    std::vector<double> Ab_global; // только на rank 0
    if (rank == 0) {
        make_diagonally_dominant_system(Ab_global, N, 42);

        if (N <= 10) {
            std::cout << "Augmented matrix [A|b] (N=" << N << "):\n";
            print_augmented(Ab_global, N);
            std::cout << "\n";
        }
    }

    // Scatterv расширенной матрицы по строкам 
    // Каждая строка имеет (N+1) элементов
    std::vector<int> sendcounts(size), senddispls(size);
    for (int p = 0; p < size; ++p) {
        sendcounts[p] = counts[p] * (N + 1);
        senddispls[p] = displs[p] * (N + 1);
    }

    // локальный блок строк
    int local_rows = counts[rank];
    std::vector<double> Ab_local(local_rows * (N + 1), 0.0);

    MPI_Scatterv(
        Ab_global.data(), sendcounts.data(), senddispls.data(), MPI_DOUBLE,
        Ab_local.data(), (int)Ab_local.size(), MPI_DOUBLE,
        0, MPI_COMM_WORLD
    );

    // Прямой ход Гаусса 
    // pivot_row — буфер для текущей опорной строки, который будет Bcast всем
    std::vector<double> pivot_row(N + 1, 0.0);

    double t0 = MPI_Wtime();

    for (int k = 0; k < N; ++k) {
        // Определяем, какой процесс владеет глобальной строкой k
        int owner = owner_of_row(k, counts, displs);

        // Процесс-владелец копирует опорную строку в pivot_row и нормализует её
        if (rank == owner) {
            int local_idx = k - displs[rank];  // локальный индекс строки k
            // Скопировали строку k в буфер pivot_row
            for (int j = 0; j < N + 1; ++j)
                pivot_row[j] = Ab_local[local_idx*(N+1) + j];

            double pivot = pivot_row[k];

            // На всякий случай защита от почти нулевого pivot
            if (std::abs(pivot) < 1e-12) {
                pivot = (pivot >= 0 ? 1e-12 : -1e-12);
            }

            // Нормализация опорной строки (чтобы ведущий элемент стал 1)
            for (int j = k; j < N + 1; ++j)
                pivot_row[j] /= pivot;

            // Обновим у себя локальную строку k тоже (чтобы локальная матрица была консистентна)
            for (int j = k; j < N + 1; ++j)
                Ab_local[local_idx*(N+1) + j] = pivot_row[j];
        }

        // Передаём опорную строку всем процессам
        MPI_Bcast(pivot_row.data(), N + 1, MPI_DOUBLE, owner, MPI_COMM_WORLD);

        // Каждый процесс зануляет элемент в столбце k для своих строк i > k
        for (int li = 0; li < local_rows; ++li) {
            int gi = displs[rank] + li; // глобальный индекс строки

            if (gi <= k) continue;      // не трогаем строки выше/равно опорной

            double factor = Ab_local[li*(N+1) + k]; // элемент, который хотим занулить

            // row_i = row_i - factor * pivot_row
            // начинаем с k, потому что левее k уже должно быть 0
            for (int j = k; j < N + 1; ++j) {
                Ab_local[li*(N+1) + j] -= factor * pivot_row[j];
            }

            // Чтобы избежать накопления мусора из-за double:
            Ab_local[li*(N+1) + k] = 0.0;
        }
    }

    double t1 = MPI_Wtime();

    // Собираем результат прямого хода на rank 0 
    if (rank == 0) Ab_global.assign(N * (N + 1), 0.0);

    MPI_Gatherv(
        Ab_local.data(), (int)Ab_local.size(), MPI_DOUBLE,
        Ab_global.data(), sendcounts.data(), senddispls.data(), MPI_DOUBLE,
        0, MPI_COMM_WORLD
    );

    // Обратный ход (делаем на rank 0) 
    if (rank == 0) {
        std::vector<double> x(N, 0.0);

        // Так как мы нормализовали опорные строки, диагональ близка к 1
        for (int i = N - 1; i >= 0; --i) {
            double sum = Ab_global[i*(N+1) + N]; // b_i
            for (int j = i + 1; j < N; ++j) {
                sum -= Ab_global[i*(N+1) + j] * x[j];
            }

            double diag = Ab_global[i*(N+1) + i];
            if (std::abs(diag) < 1e-12) diag = 1e-12;

            x[i] = sum / diag;
        }

        std::cout << "Solution x (first 10 values):\n";
        for (int i = 0; i < std::min(N, 10); ++i) {
            std::cout << "x[" << i << "] = " << std::setprecision(8) << x[i] << "\n";
        }

        std::cout << "\nForward elimination time (MPI_Wtime): "
                  << std::setprecision(6) << (t1 - t0) << " seconds\n";
    }

    MPI_Finalize();
    return 0;
}
