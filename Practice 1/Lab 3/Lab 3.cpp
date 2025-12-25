#include <iostream>
#include <cstdlib>   // для rand() и srand()
#include <ctime>     // для time()
#include <iomanip>   // для форматированного вывода

#ifdef _OPENMP
#include <omp.h>     // для работы с OpenMP
#endif

using namespace std;

// Function for sequential (non-parallel) average calculation
double averageSequential(const int* array, int size) {
    long long sum = 0;
    // используем long long, чтобы избежать переполнения при сложении

    for (int i = 0; i < size; i++) {
        sum += array[i]; // добавляем каждый элемент массива
    }

    // возвращаем среднее значение
    return (double)sum / size;
}

// Function for parallel average calculation using OpenMP
double averageParallelOpenMP(const int* array, int size) {
    long long sum = 0;

    // каждый поток считает свою часть суммы,
    // reduction объединяет все частичные суммы в одну
    #ifdef _OPENMP
    #pragma omp parallel for reduction(+:sum)
    #endif
    for (int i = 0; i < size; i++) {
        sum += array[i];
    }

    return (double)sum / size;
}

int main() {
    int size;
    cout << "Enter array size: ";
    cin >> size;

    // проверяем, чтобы размер массива был больше нуля
    if (size <= 0) {
        cout << "Array size must be greater than zero." << endl;
        return 0;
    }

    // 1) создаём динамический массив с помощью указателя
    int* array = new int[size];

    // инициализируем генератор случайных чисел
    srand(time(nullptr));

    // заполняем массив случайными числами от 0 до 99
    for (int i = 0; i < size; i++) {
        array[i] = rand() % 100;
    }

    // выводим несколько первых элементов массива для проверки
    cout << "First elements of the array: ";
    for (int i = 0; i < (size < 10 ? size : 10); i++) {
        cout << array[i] << " ";
    }
    cout << endl;

    // 2) вычисляем среднее значение обычным способом
    double averageSequentialResult = averageSequential(array, size);

    // 3) вычисляем среднее значение параллельно с OpenMP
    double averageParallelResult = averageParallelOpenMP(array, size);

    cout << fixed << setprecision(3);
    cout << "Average value (sequential): " << averageSequentialResult << endl;
    cout << "Average value (parallel with OpenMP): " << averageParallelResult << endl;

    #ifdef _OPENMP
    cout << "OpenMP is enabled. Maximum number of threads: "
         << omp_get_max_threads() << endl;
    #else
    cout << "OpenMP is not enabled during compilation." << endl;
    #endif

    // 4) освобождаем выделенную динамическую память
    delete[] array;
    array = nullptr; // чтобы случайно не использовать после удаления

    return 0;
}
