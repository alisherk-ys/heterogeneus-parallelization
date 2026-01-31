#include <mpi.h>
#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include <algorithm>
#include <cmath>

// "Бесконечность" для отсутствующих ребер
static constexpr double INF = 1e15;

// Печать матрицы (полезно только для маленького N)
static void print_matrix(const std::vector<double>& M, int N) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            double v = M[i * N + j];
            if (v > INF / 2) std::cout << std::setw(6) << "INF";
            else             std::cout << std::setw(6) << (int)std::round(v);
        }
        std::cout << "\n";
    }
}

// Генерация случайного графа (матрица смежности)
// probEdge — вероятность ребра i->j
// weights 1..W
static void generate_graph(std::vector<double>& G, int N, double probEdge, int W, unsigned seed = 42) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> p(0.0, 1.0);
    std::uniform_int_distribution<int> w(1, W);

    G.assign(N * N, INF);

    for (int i = 0; i < N; ++i) {
        G[i * N + i] = 0.0; // диагональ = 0
        for (int j = 0; j < N; ++j) {
            if (i == j) continue;
            if (p(gen) < probEdge) {
                G[i * N + j] = (double)w(gen);
            }
        }
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Параметры 
    // argv[1] = N (размер графа)
    // argv[2] = probEdge (вероятность ребра)
    // argv[3] = maxWeight (максимальный вес ребра)
    int N = 8;                 // маленький по умолчанию (для отладки/печати)
    double probEdge = 0.30;    // плотность ребер
    int maxWeight = 20;        // максимум веса

    if (argc >= 2) N = std::max(2, std::atoi(argv[1]));
    if (argc >= 3) probEdge = std::max(0.0, std::min(1.0, std::atof(argv[2])));
    if (argc >= 4) maxWeight = std::max(1, std::atoi(argv[3]));

    // Распределение строк 
    // counts[p] — сколько строк у процесса p
    // displs[p] — с какой глобальной строки начинается блок процесса p
    std::vector<int> counts(size), displs(size);
    int base = N / size;
    int rem  = N % size;

    for (int p = 0; p < size; ++p) counts[p] = base + (p < rem ? 1 : 0);
    displs[0] = 0;
    for (int p = 1; p < size; ++p) displs[p] = displs[p - 1] + counts[p - 1];

    // Для Scatterv/Allgatherv нужны количества элементов (а не строк)
    std::vector<int> sendcounts(size), senddispls(size);
    for (int p = 0; p < size; ++p) {
        sendcounts[p] = counts[p] * N;   // строк * N столбцов
        senddispls[p] = displs[p] * N;   // смещение по элементам
    }

    // rank 0 создаёт полный граф G 
    std::vector<double> G_full;
    if (rank == 0) {
        generate_graph(G_full, N, probEdge, maxWeight, 42);

        if (N <= 12) {
            std::cout << "Initial adjacency matrix (N=" << N << "):\n";
            print_matrix(G_full, N);
            std::cout << "\n";
        }
    }

    // Каждый процесс хранит свой блок строк 
    int local_rows = counts[rank];
    std::vector<double> G_local(local_rows * N, INF);

    // Рассылаем строки матрицы по процессам 
    // В задании указан MPI_Scatter, но используем Scatterv (корректно при любом np)
    MPI_Scatterv(
        G_full.data(), sendcounts.data(), senddispls.data(), MPI_DOUBLE,
        G_local.data(), (int)G_local.size(), MPI_DOUBLE,
        0, MPI_COMM_WORLD
    );

    // На всех процессах будет копия полной матрицы (нужна для чтения строки k) 
    std::vector<double> G_all(N * N, INF);

    // Сразу соберем стартовую матрицу на всех (один Allgatherv)
    MPI_Allgatherv(
        G_local.data(), (int)G_local.size(), MPI_DOUBLE,
        G_all.data(), sendcounts.data(), senddispls.data(), MPI_DOUBLE,
        MPI_COMM_WORLD
    );

    double t0 = MPI_Wtime();

    // Алгоритм Флойда–Уоршелла 
    // dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
    for (int k = 0; k < N; ++k) {
        // Строка k доступна всем через G_all (она собрана после предыдущего шага)
        const double* row_k = &G_all[k * N];

        // Обновляем только локальные строки (которые принадлежат этому процессу)
        for (int li = 0; li < local_rows; ++li) {
            int gi = displs[rank] + li;      // глобальный индекс строки
            double dik = G_local[li * N + k]; // dist[i][k] (локально хранится)

            // Если i->k недостижимо, смысла обновлять нет
            if (dik > INF / 2) continue;

            // Для каждого j пробуем улучшить путь через k
            for (int j = 0; j < N; ++j) {
                double dkj = row_k[j];
                if (dkj > INF / 2) continue;

                double cand = dik + dkj;
                double &dij = G_local[li * N + j];

                if (cand < dij) dij = cand;
            }
        }

        // После обновления - обмен между процессами: собираем новую матрицу на всех (это и есть требование MPI_Allgather / MPI_Allgatherv)
        MPI_Allgatherv(
            G_local.data(), (int)G_local.size(), MPI_DOUBLE,
            G_all.data(), sendcounts.data(), senddispls.data(), MPI_DOUBLE,
            MPI_COMM_WORLD
        );
    }

    double t1 = MPI_Wtime();

    // После завершения G_all содержит финальные расстояния на всех процессах.
    // По заданию выводим на rank 0.
    if (rank == 0) {
        std::cout << "Floyd–Warshall finished.\n";
        std::cout << "Execution time (MPI_Wtime): " << std::setprecision(6) << (t1 - t0) << " seconds\n\n";

        if (N <= 12) {
            std::cout << "All-pairs shortest paths matrix:\n";
            print_matrix(G_all, N);
            std::cout << "\n";
        } else {
            // Для больших N печатать матрицу нельзя — слишком много
            // Выведем несколько значений как "контроль"
            std::cout << "Sample distances:\n";
            for (int i = 0; i < std::min(N, 5); ++i) {
                for (int j = 0; j < std::min(N, 5); ++j) {
                    double v = G_all[i * N + j];
                    if (v > INF / 2) std::cout << "INF ";
                    else std::cout << (long long)std::llround(v) << " ";
                }
                std::cout << "\n";
            }
        }
    }

    MPI_Finalize();
    return 0;
}
