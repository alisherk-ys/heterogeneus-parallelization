#include <mpi.h>        
#include <iostream>     
#include <vector>       
#include <random>       
#include <iomanip>      
#include <algorithm>    
#include <cmath>       
#include <cstring>      

// Тайминги этапов
struct Times {
    double gen = 0;     // генерация данных
    double comp = 0;    // локальные вычисления
    double comm = 0;    // коммуникации MPI
    double total = 0;   // полное время
};

// Локальная агрегация: сумма, минимум, максимум
static void local_aggregate(const std::vector<double>& a,
                            double& sum, double& mn, double& mx)
{
    sum = 0.0;
    mn = a.empty() ? 0.0 : a[0];
    mx = a.empty() ? 0.0 : a[0];
    for (double x : a) {
        sum += x;
        mn = std::min(mn, x);
        mx = std::max(mx, x);
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank=0, size=1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Аргументы:
    // mode: strong / weak
    // rmode: reduce / allreduce
    // N: размер данных
    // iters: число прогонов
    std::string mode  = (argc >= 2) ? argv[1] : "strong";
    std::string rmode = (argc >= 3) ? argv[2] : "reduce";
    long long N_arg   = (argc >= 4) ? std::atoll(argv[3]) : 10'000'000LL;
    int iters         = (argc >= 5) ? std::atoi(argv[4]) : 20;

    // Определение global_N и local_N
    long long local_N = 0, global_N = 0;
    if (mode == "weak") {
        local_N = std::max(1LL, N_arg);
        global_N = local_N * size;
    } else {
        global_N = std::max(1LL, N_arg);
        local_N = (global_N + size - 1) / size;
    }

    // Диапазон данных текущего процесса
    long long start = (long long)rank *
        ((mode=="weak") ? local_N : (global_N + size - 1) / size);
    long long end = std::min(global_N, start + local_N);
    long long real_local_N = std::max(0LL, end - start);

    // Локальный буфер
    std::vector<double> a((size_t)real_local_N);

    // Один прогон
    auto run_once = [&](Times& T){
        MPI_Barrier(MPI_COMM_WORLD);
        double t0 = MPI_Wtime();

        // 1) Генерация данных
        double tg0 = MPI_Wtime();
        std::mt19937_64 gen(1234 + rank);
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        for (long long i = 0; i < real_local_N; ++i)
            a[(size_t)i] = dist(gen);
        double tg1 = MPI_Wtime();

        // 2) Локальные вычисления
        double tc0 = MPI_Wtime();
        double lsum=0, lmin=0, lmax=0;
        local_aggregate(a, lsum, lmin, lmax);
        double tc1 = MPI_Wtime();

        // 3) MPI-коммуникации
        double tcomm0 = MPI_Wtime();
        double gsum=0, gmin=0, gmax=0;

        if (rmode == "allreduce") {
            MPI_Allreduce(&lsum, &gsum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(&lmin, &gmin, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
            MPI_Allreduce(&lmax, &gmax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        } else {
            MPI_Reduce(&lsum, &gsum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            MPI_Reduce(&lmin, &gmin, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
            MPI_Reduce(&lmax, &gmax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        }

        double tcomm1 = MPI_Wtime();

        MPI_Barrier(MPI_COMM_WORLD);
        double t1 = MPI_Wtime();

        // Запись таймингов
        T.gen   = tg1 - tg0;
        T.comp  = tc1 - tc0;
        T.comm  = tcomm1 - tcomm0;
        T.total = t1 - t0;

        if (rank == 0) {
            (void)gsum; (void)gmin; (void)gmax;
        }
    };

    // Усреднение по итерациям
    Times local_avg{};
    for (int i = 0; i < iters; ++i) {
        Times t{};
        run_once(t);
        local_avg.gen   += t.gen;
        local_avg.comp  += t.comp;
        local_avg.comm  += t.comm;
        local_avg.total += t.total;
    }
    local_avg.gen   /= iters;
    local_avg.comp  /= iters;
    local_avg.comm  /= iters;
    local_avg.total /= iters;

    // Максимум по процессам
    auto max_over_ranks = [&](double x){
        double g=0;
        MPI_Allreduce(&x, &g, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        return g;
    };

    double gen_max   = max_over_ranks(local_avg.gen);
    double comp_max  = max_over_ranks(local_avg.comp);
    double comm_max  = max_over_ranks(local_avg.comm);
    double total_max = max_over_ranks(local_avg.total);

    // Вывод результатов
    if (rank == 0) {
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "MPI Scaling\n";
        std::cout << "Mode = " << mode << " | Comm = " << rmode
                  << " | iters = " << iters << "\n";
        std::cout << "Processes = " << size
                  << " | Global N = " << global_N
                  << " | Local N(avg) ~ " << real_local_N << "\n\n";
        std::cout << "Max over ranks (seconds):\n";
        std::cout << "  Data gen   = " << gen_max   << "\n";
        std::cout << "  Compute    = " << comp_max  << "\n";
        std::cout << "  Comm       = " << comm_max  << "\n";
        std::cout << "  Total      = " << total_max << "\n";
        std::cout << "\nComm share (Comm/Total) = "
                  << (total_max > 0 ? (comm_max / total_max) : 0) << "\n";
    }

    MPI_Finalize();
    return 0;
}
