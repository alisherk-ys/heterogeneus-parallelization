#include <iostream>
#include <vector>
#include <omp.h>        // Библиотека OpenMP
#include <chrono>       // Для замера времени

int main() {
    const int N = 1'000'000;   // Размер массива

    // Создание массива 
    std::vector<double> data(N);

    // Инициализация массива (например, значениями индексов)
    for (int i = 0; i < N; ++i) {
        data[i] = static_cast<double>(i);
    }

    // Замер времени 
    // Используем high_resolution_clock для замера времени
    auto start_time = std::chrono::high_resolution_clock::now();

    // Параллельная обработка OpenMP 
    // Каждый элемент массива умножается на 2
    // pragma omp parallel for — распределяет цикл между потоками
    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        data[i] *= 2.0;
    }

    // Окончание замера времени
    auto end_time = std::chrono::high_resolution_clock::now();

    // Вычисляем длительность в миллисекундах
    std::chrono::duration<double, std::milli> elapsed = end_time - start_time;

    // Вывод результатов 
    std::cout << "Array size: " << N << std::endl;
    std::cout << "CPU processing time (OpenMP): "
              << elapsed.count() << " ms" << std::endl;

    // Проверка корректности (необязательно, но полезно)
    std::cout << "Sample values:" << std::endl;
    std::cout << "data[0] = " << data[0] << std::endl;
    std::cout << "data[N/2] = " << data[N / 2] << std::endl;
    std::cout << "data[N-1] = " << data[N - 1] << std::endl;

    return 0;
}
