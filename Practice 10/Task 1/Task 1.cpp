#include <omp.h>        
#include <iostream>    
#include <vector>       
#include <random>       
#include <iomanip>      
#include <cmath>        
#include <algorithm>    

// Структура для хранения статистических показателей
struct Stats {
    double sum = 0.0;   // сумма элементов
    double mean = 0.0;  // среднее значение
    double var  = 0.0;  // дисперсия (population)
};

// Последовательная версия:
// вычисляет сумму, среднее и дисперсию
Stats stats_sequential(const std::vector<double>& a, double& t_compute) {
    // Засекаем время начала вычислений
    double t0 = omp_get_wtime();

    double sum = 0.0;
    double sumsq = 0.0;

    // Последовательный проход по массиву
    for (size_t i = 0; i < a.size(); ++i) {
        sum   += a[i];        // накапливаем сумму
        sumsq += a[i] * a[i]; // накапливаем сумму квадратов
    }

    // Вычисление среднего
    double mean = sum / (double)a.size();

    // Вычисление дисперсии: E[x^2] - (E[x])^2
    double var  = sumsq / (double)a.size() - mean * mean;

    // Засекаем время окончания
    double t1 = omp_get_wtime();
    t_compute = (t1 - t0); // чистое время вычислений

    return {sum, mean, var};
}


// Параллельная версия:
// используется reduction для sum и sumsq
Stats stats_parallel(const std::vector<double>& a, int threads, double& t_compute) {
    // Время начала вычислений
    double t0 = omp_get_wtime();

    double sum = 0.0;
    double sumsq = 0.0;

    // Параллельный цикл OpenMP
    // num_threads(threads) — количество потоков
    // reduction(+:sum,sumsq) — корректное суммирование между потоками
    // schedule(static) — равномерное распределение итераций
    #pragma omp parallel for num_threads(threads) reduction(+:sum,sumsq) schedule(static)
    for (long long i = 0; i < (long long)a.size(); ++i) {
        double x = a[(size_t)i];
        sum   += x;
        sumsq += x * x;
    }

    // Среднее значение
    double mean = sum / (double)a.size();

    // Дисперсия
    double var  = sumsq / (double)a.size() - mean * mean;

    // Время окончания
    double t1 = omp_get_wtime();
    t_compute = (t1 - t0);

    return {sum, mean, var};
}

// Оценка последовательной доли по закону Амдала
// S = 1 / ( s + (1-s)/p )
// Выражаем s (последовательную часть)
double amdahl_serial_fraction(double speedup, int p) {
    if (p <= 1) return 1.0; // если 1 поток - всё последовательно

    double invS = 1.0 / speedup;
    double invP = 1.0 / (double)p;

    return (invS - invP) / (1.0 - invP);
}

int main(int argc, char** argv) {
    // Размер массива:
    // по умолчанию 10 млн элементов
    const size_t N = (argc >= 2)
        ? (size_t)std::max(1LL, std::stoll(argv[1]))
        : 10'000'000;

    // Максимальное количество потоков
    const int max_threads = (argc >= 3)
        ? std::max(1, std::stoi(argv[2]))
        : std::min(16, omp_get_max_threads());

    std::cout << "Practical Work #10 (OpenMP): sum/mean/variance\n";
    std::cout << "N = " << N << ", max_threads = " << max_threads << "\n\n";


    // 1) Инициализация данных
    double t_init0 = omp_get_wtime();

    std::vector<double> a(N);

    // Фиксированный seed для воспроизводимости результатов
    std::mt19937_64 gen(123);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    // Заполнение массива случайными числами
    for (size_t i = 0; i < N; ++i)
        a[i] = dist(gen);

    double t_init1 = omp_get_wtime();
    double t_init = (t_init1 - t_init0);


    // 2) Последовательная версия (baseline)
    double t_seq = 0.0;
    Stats s_seq = stats_sequential(a, t_seq);

    // 3) Параллельная версия:
    // тестируем разные количества потоков
    std::vector<int> thread_list;

    // Формируем список: 1, 2, 4, 8, ...
    for (int t = 1; t <= max_threads; t *= 2)
        thread_list.push_back(t);

    // Добавляем max_threads, если он не степень двойки
    if (thread_list.back() != max_threads)
        thread_list.push_back(max_threads);

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Init time (s)          = " << t_init << "\n";
    std::cout << "Sequential compute (s) = " << t_seq << "\n";
    std::cout << "Seq result: sum=" << s_seq.sum
              << " mean=" << s_seq.mean
              << " var=" << s_seq.var << "\n\n";

    std::cout << "Threads | Par time (s) | Speedup | Eff(%) | Amdahl serial s | Parallel part (1-s)\n";

    // Прогоны для каждого числа потоков
    for (int t : thread_list) {
        double t_par = 0.0;
        Stats s_par = stats_parallel(a, t, t_par);

        // Проверка корректности результатов
        double max_err = std::max({
            std::fabs(s_par.sum  - s_seq.sum),
            std::fabs(s_par.mean - s_seq.mean),
            std::fabs(s_par.var  - s_seq.var)
        });

        bool ok = max_err < 1e-6;

        // Ускорение
        double speedup = t_seq / t_par;

        // Эффективность (%)
        double eff = (speedup / (double)t) * 100.0;

        // Последовательная доля по Амдалу
        double s = amdahl_serial_fraction(speedup, t);
        s = std::clamp(s, 0.0, 1.0);

        // Вывод результатов
        std::cout << std::setw(7)  << t << " | "
                  << std::setw(10) << t_par << " | "
                  << std::setw(7)  << speedup << " | "
                  << std::setw(6)  << eff << " | "
                  << std::setw(15) << s << " | "
                  << std::setw(18) << (1.0 - s)
                  << (ok ? "" : "  (WARN: mismatch)") << "\n";
    }

    return 0;
}
