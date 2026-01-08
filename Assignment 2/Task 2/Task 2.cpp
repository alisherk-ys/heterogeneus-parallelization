#include <iostream>
#include <vector>
#include <random>
#include <limits>
#include <chrono>
#include <omp.h>

using namespace std;

int main() {
    // Размер массива по заданию
    const int N = 10000;

    // ----------------------------
    // 1) Создание массива и заполнение
    // ----------------------------
    vector<int> arr(N);

    // Генератор случайных чисел
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<int> dist(1, 100000);

    // Заполняем массив случайными значениями
    for (int i = 0; i < N; i++) {
        arr[i] = dist(gen);
    }

    // ----------------------------
    // 2) Последовательный поиск min/max
    // ----------------------------
    int seqMin = numeric_limits<int>::max();
    int seqMax = numeric_limits<int>::min();

    // Засекаем время (chrono подходит для CPU-кода)
    auto startSeq = chrono::high_resolution_clock::now();

    // Проходим по всем элементам и обновляем min/max
    for (int i = 0; i < N; i++) {
        if (arr[i] < seqMin) seqMin = arr[i];
        if (arr[i] > seqMax) seqMax = arr[i];
    }

    auto endSeq = chrono::high_resolution_clock::now();
    double seqMs = chrono::duration<double, milli>(endSeq - startSeq).count();

    // ----------------------------
    // 3) Параллельный поиск min/max (OpenMP)
    // ----------------------------
    // Для параллельной версии используем reduction(min:...) и reduction(max:...),
    // чтобы OpenMP корректно объединил локальные значения потоков в общий результат.
    int parMin = numeric_limits<int>::max();
    int parMax = numeric_limits<int>::min();

    // omp_get_wtime() удобно использовать для замеров времени в OpenMP
    double startPar = omp_get_wtime();

    #pragma omp parallel for reduction(min:parMin) reduction(max:parMax)
    for (int i = 0; i < N; i++) {
        // Каждый поток обрабатывает свою часть диапазона
        if (arr[i] < parMin) parMin = arr[i];
        if (arr[i] > parMax) parMax = arr[i];
    }

    double endPar = omp_get_wtime();
    double parMs = (endPar - startPar) * 1000.0;

    // ----------------------------
    // 4) Вывод результатов и выводы
    // ----------------------------
    cout << "Array size: " << N << "\n\n";

    cout << "Sequential result:\n";
    cout << "  min = " << seqMin << "\n";
    cout << "  max = " << seqMax << "\n";
    cout << "  time = " << seqMs << " ms\n\n";

    cout << "OpenMP parallel result:\n";
    cout << "  min = " << parMin << "\n";
    cout << "  max = " << parMax << "\n";
    cout << "  time = " << parMs << " ms\n\n";

    // Проверка, что результаты совпадают
    if (seqMin == parMin && seqMax == parMax) {
        cout << "Check: OK (results match)\n";
    } else {
        cout << "Check: FAIL (results differ)\n";
    }

    return 0;
}

