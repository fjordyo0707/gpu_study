#include <cuda_runtime.h>

#include <cmath>
#include <iostream>
#include <vector>
#include <algorithm>

__global__ void vector_add_strided(const float *a,
                                   const float *b,
                                   float *c,
                                   int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    int idx = i * 2;

    if (idx < n)
    {
        c[idx] = a[idx] + b[idx];
    }
}

int main(int argc, char **argv)
{
    const int N = 1 << 24;
    const int iterations = 100;

    size_t bytes = N * sizeof(float);

    std::vector<float> h_a(N, 1.0f);
    std::vector<float> h_b(N, 2.0f);
    std::vector<float> h_c(N);

    float *d_a, *d_b, *d_c;

    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice);

    int threads = 256;

    if (argc > 1)
    {
        threads = std::stoi(argv[1]);
    }
    int blocks = (N + threads - 1) / threads;

    // Warm-up
    vector_add_strided<<<blocks, threads>>>(d_a, d_b, d_c, N);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    std::vector<float> times(iterations);

    for (int i = 0; i < iterations; i++)
    {
        cudaEventRecord(start);

        vector_add_strided<<<blocks, threads>>>(d_a, d_b, d_c, N);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        cudaEventElapsedTime(&times[i], start, stop);
    }

    // Verify result
    cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost);

    bool correct = true;

    for (int i = 0; i < N; i += 2)
    {
        if (std::fabs(h_c[i] - 3.0f) > 1e-5f)
        {
            correct = false;
            break;
        }
    }

    float sum = 0.0f;

    for (float t : times)
    {
        sum += t;
    }

    float average = sum / iterations;

    float minimum = *std::min_element(times.begin(), times.end());
    float maximum = *std::max_element(times.begin(), times.end());

    double total_bytes = 3.0 * bytes;

    double bandwidth_gb_s =
        total_bytes / (average / 1000.0) / 1e9;

    std::cout << "N                  = " << N << "\n";
    std::cout << "Threads per block   = " << threads << "\n";
    std::cout << "Blocks              = " << blocks << "\n";
    std::cout << "Iterations          = " << iterations << "\n";

    std::cout << "\n";

    std::cout << "Min kernel time     = "
              << minimum << " ms\n";

    std::cout << "Max kernel time     = "
              << maximum << " ms\n";

    std::cout << "Average kernel time = "
              << average << " ms\n";

    std::cout << "Effective bandwidth = "
              << bandwidth_gb_s << " GB/s\n";

    std::cout << "Result              = "
              << (correct ? "PASS" : "FAIL") << "\n";

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return correct ? 0 : 1;
}