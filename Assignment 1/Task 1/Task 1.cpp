#include <iostream>
#include <random>

using namespace std;

int main() {
    // Размер массива по заданию
    const int N = 50000;

    // Выделяем память под массив из 50 000 чисел
    // Используем динамическую память, потому что размер задаётся в программе
    int* arr = new int[N];

    // Настраиваем генерацию случайных чисел
    random_device rd;      // источник начального значения
    mt19937 gen(rd());     // генератор
    uniform_int_distribution<int> dist(1, 100); // числа от 1 до 100

    // Заполняем массив случайными числами
    for (int i = 0; i < N; i++) {
        arr[i] = dist(gen);
    }

    // Считаем сумму элементов массива
    // Используем long long, чтобы сумма не переполнилась
    long long sum = 0;
    for (int i = 0; i < N; i++) {
        sum += arr[i];
    }

    // Вычисляем среднее значение
    double average = (double)sum / N;

    // Выводим результат
    cout << "Task 1\n";
    cout << "Average value = " << average << endl;

    // Освобождаем выделенную память
    delete[] arr;

    return 0;
}
