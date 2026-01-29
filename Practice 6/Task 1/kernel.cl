__kernel void vector_add(__global const float* A,
                         __global const float* B,
                         __global float* C,
                         const int n) // добавили n, чтобы безопасно обрабатывать хвост
{
    int id = get_global_id(0);  // глобальный индекс потока
    if (id < n) {
        C[id] = A[id] + B[id];
    }
}
