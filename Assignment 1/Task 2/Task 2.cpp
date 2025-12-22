#include <iostream>
#include <random>
#include <omp.h>

using namespace std;

int main() {
    // Размер массива
    const int N = 1000000;

    // Выделяем память под массив
    int* arr = new int[N];

    // Генерация случайных чисел от 1 до 100
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<int> dist(1, 100);

    // Заполняем массив
    for (int i = 0; i < N; i++) {
        arr[i] = dist(gen);
    }

    // Начальные значения min и max берём из первого элемента
    int minVal = arr[0];
    int maxVal = arr[0];

    // Засекаем время начала выполнения
    double start = omp_get_wtime();

    // Последовательно проходим по массиву
    for (int i = 1; i < N; i++) {
        if (arr[i] < minVal)
            minVal = arr[i];

        if (arr[i] > maxVal)
            maxVal = arr[i];
    }

    // Засекаем время окончания
    double end = omp_get_wtime();

    // Вывод результатов
    cout << "Min = " << minVal << endl;
    cout << "Max = " << maxVal << endl;
    cout << "Time = " << end - start << " seconds" << endl;

    // Освобождаем память
    delete[] arr;

    return 0;
}
