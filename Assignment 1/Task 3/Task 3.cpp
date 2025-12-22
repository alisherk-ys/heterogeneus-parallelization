#include <iostream>
#include <random>
#include <limits>
#include <omp.h>

using namespace std;

int main() {
    const int N = 1000000;

    int* arr = new int[N];

    // Генерация случайных чисел
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<int> dist(1, 100);

    for (int i = 0; i < N; i++) {
        arr[i] = dist(gen);
    }

    // -------- Последовательная версия --------
    int minSeq = arr[0];
    int maxSeq = arr[0];

    double t1 = omp_get_wtime();

    for (int i = 1; i < N; i++) {
        if (arr[i] < minSeq) minSeq = arr[i];
        if (arr[i] > maxSeq) maxSeq = arr[i];
    }

    double t2 = omp_get_wtime();

    // -------- Параллельная версия --------
    int minPar = numeric_limits<int>::max();
    int maxPar = numeric_limits<int>::min();

    double t3 = omp_get_wtime();

    #pragma omp parallel
    {
        // Локальные min и max для каждого потока
        int localMin = numeric_limits<int>::max();
        int localMax = numeric_limits<int>::min();

        // Каждый поток обрабатывает свою часть массива
        #pragma omp for
        for (int i = 0; i < N; i++) {
            if (arr[i] < localMin) localMin = arr[i];
            if (arr[i] > localMax) localMax = arr[i];
        }

        // Объединяем результаты потоков
        #pragma omp critical
        {
            if (localMin < minPar) minPar = localMin;
            if (localMax > maxPar) maxPar = localMax;
        }
    }

    double t4 = omp_get_wtime();

    // Вывод результатов
    cout << "Sequential:\n";
    cout << "Min = " << minSeq << ", Max = " << maxSeq << endl;
    cout << "Time = " << t2 - t1 << " seconds\n\n";

    cout << "Parallel:\n";
    cout << "Min = " << minPar << ", Max = " << maxPar << endl;
    cout << "Time = " << t4 - t3 << " seconds\n\n";

    cout << "Speedup = " << (t2 - t1) / (t4 - t3) << endl;

    delete[] arr;
    return 0;
}
