#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <omp.h>

using namespace std;

// Проверка, что массив отсортирован по возрастанию
bool isSorted(const vector<int>& a) {
    for (int i = 1; i < (int)a.size(); i++) {
        if (a[i - 1] > a[i]) return false;
    }
    return true;
}

// ----------------------------
// Последовательная сортировка выбором
// ----------------------------
void selectionSortSequential(vector<int>& a) {
    int n = (int)a.size();

    // i — граница между отсортированной и неотсортированной частью
    for (int i = 0; i < n - 1; i++) {
        int minIndex = i;

        // Находим индекс минимального элемента в диапазоне [i..n-1]
        for (int j = i + 1; j < n; j++) {
            if (a[j] < a[minIndex]) {
                minIndex = j;
            }
        }

        // Ставим найденный минимум на позицию i
        swap(a[i], a[minIndex]);
    }
}

// ----------------------------
// Параллельная сортировка выбором
// ----------------------------
// Важно: весь алгоритм целиком распараллелить сложно, потому что шаги i зависят друг от друга.
// Поэтому здесь распараллеливается только поиск минимума на текущем шаге i.
void selectionSortParallel(vector<int>& a) {
    int n = (int)a.size();

    for (int i = 0; i < n - 1; i++) {
        int globalMinValue = a[i];
        int globalMinIndex = i;

        // Создаём параллельную область
        #pragma omp parallel
        {
            // Локальные значения для каждого потока
            int localMinValue = globalMinValue;
            int localMinIndex = globalMinIndex;

            // Каждый поток ищет минимум в своей части диапазона
            #pragma omp for nowait
            for (int j = i + 1; j < n; j++) {
                if (a[j] < localMinValue) {
                    localMinValue = a[j];
                    localMinIndex = j;
                }
            }

            // Объединяем результаты потоков в общий минимум
            // critical нужен, чтобы два потока не записали глобальный минимум одновременно
            #pragma omp critical
            {
                if (localMinValue < globalMinValue) {
                    globalMinValue = localMinValue;
                    globalMinIndex = localMinIndex;
                }
            }
        }

        // После нахождения минимального элемента выполняем обмен
        swap(a[i], a[globalMinIndex]);
    }
}

int main() {
    // Размеры массивов по заданию
    vector<int> sizes = {1000, 10000};

    // Генератор случайных чисел
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<int> dist(1, 100000);

    for (int N : sizes) {
        // Создаём исходный массив
        vector<int> base(N);
        for (int i = 0; i < N; i++) base[i] = dist(gen);

        // Делаем две копии: для последовательной и параллельной сортировки
        vector<int> aSeq = base;
        vector<int> aPar = base;

        // ----------------------------
        // Последовательная версия (замер времени)
        // ----------------------------
        auto startSeq = chrono::high_resolution_clock::now();
        selectionSortSequential(aSeq);
        auto endSeq = chrono::high_resolution_clock::now();
        double seqMs = chrono::duration<double, milli>(endSeq - startSeq).count();

        // ----------------------------
        // Параллельная версия (замер времени)
        // ----------------------------
        double startPar = omp_get_wtime();
        selectionSortParallel(aPar);
        double endPar = omp_get_wtime();
        double parMs = (endPar - startPar) * 1000.0;

        // ----------------------------
        // Вывод результатов
        // ----------------------------
        cout << "\nArray size: " << N << "\n";
        cout << "Sequential time: " << seqMs << " ms\n";
        cout << "Parallel time:   " << parMs << " ms\n";

        cout << "Sorted check (sequential): " << (isSorted(aSeq) ? "OK" : "FAIL") << "\n";
        cout << "Sorted check (parallel):   " << (isSorted(aPar) ? "OK" : "FAIL") << "\n";

    }

    return 0;
}
