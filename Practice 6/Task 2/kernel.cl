__kernel void matmul_basic(__global const float* A,
                           __global const float* B,
                           __global float* C,
                           const int N, const int M, const int K)
{
    // row = i, col = j
    int row = get_global_id(0);
    int col = get_global_id(1);

    if (row < N && col < K) {
        float sum = 0.0f;
        for (int t = 0; t < M; ++t) {
            sum += A[row * M + t] * B[t * K + col];
        }
        C[row * K + col] = sum;
    }
}
