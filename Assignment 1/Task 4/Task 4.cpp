#include <iostream>
#include <random>
#include <omp.h>

using namespace std;

int main() {
    const int N = 5000000;

    // Выделяем память под массив
    double* arr = new double[N];

    // Генерация случайных чисел
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<int> dist(1, 100);

    // Заполняем массив
    for (int i = 0; i < N; i++) {
        arr[i] = dist(gen);
    }

    // Последовательный подсчёт среднего
    double startSeq = omp_get_wtime();

    double sumSeq = 0;
    for (int i = 0; i < N; i++) {
        sumSeq += arr[i];
    }

    double avgSeq = sumSeq / N;

    double endSeq = omp_get_wtime();

    // Параллельный подсчёт среднего
    double sumPar = 0;

    double startPar = omp_get_wtime();

    // reduction нужен, чтобы каждый поток считал свою сумму,
    // а потом OpenMP аккуратно сложил их вместе
    #pragma omp parallel for reduction(+:sumPar)
    for (int i = 0; i < N; i++) {
        sumPar += arr[i];
    }

    double avgPar = sumPar / N;

    double endPar = omp_get_wtime();

    // Вывод результатов
    cout << "Sequential:\n";
    cout << "Average = " << avgSeq << endl;
    cout << "Time = " << endSeq - startSeq << " seconds\n\n";

    cout << "Parallel:\n";
    cout << "Average = " << avgPar << endl;
    cout << "Time = " << endPar - startPar << " seconds\n\n";

    cout << "Speedup = " << (endSeq - startSeq) / (endPar - startPar) << endl;

    delete[] arr;
    return 0;
}
