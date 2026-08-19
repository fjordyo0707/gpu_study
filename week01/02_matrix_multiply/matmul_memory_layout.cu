#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#define CHECK_CUDA(call)                                                       \
    do                                                                         \
    {                                                                          \
        cudaError_t status = (call);                                           \
        if (status != cudaSuccess)                                             \
        {                                                                      \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__      \
                      << ": " << cudaGetErrorString(status) << "\n";          \
            return 1;                                                          \
        }                                                                      \
    } while (0)

__global__ void matmul_normal_b(const float *a,
                                const float *b,
                                float *c,
                                int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n)
    {
        float sum = 0.0f;

        for (int k = 0; k < n; k++)
        {
            sum += a[row * n + k] * b[k * n + col];
        }

        c[row * n + col] = sum;
    }
}

__global__ void matmul_transposed_b(const float *a,
                                    const float *b_t,
                                    float *c,
                                    int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n)
    {
        float sum = 0.0f;

        for (int k = 0; k < n; k++)
        {
            sum += a[row * n + k] * b_t[col * n + k];
        }

        c[row * n + col] = sum;
    }
}

int parse_positive_int(const char *value, const char *name)
{
    int parsed = std::stoi(value);

    if (parsed <= 0)
    {
        throw std::invalid_argument(std::string(name) + " must be positive");
    }

    return parsed;
}

bool verify_result(const std::vector<float> &values, float expected)
{
    for (float value : values)
    {
        if (std::fabs(value - expected) > 1e-3f)
        {
            return false;
        }
    }

    return true;
}

int main(int argc, char **argv)
{
    int n = 1024;
    int block_dim = 16;
    int iterations = 20;
    const int warmups = 3;

    try
    {
        if (argc > 1)
        {
            n = parse_positive_int(argv[1], "matrix_size");
        }

        if (argc > 2)
        {
            block_dim = parse_positive_int(argv[2], "block_dim");
        }

        if (argc > 3)
        {
            iterations = parse_positive_int(argv[3], "iterations");
        }

        if (block_dim > 1024 / block_dim)
        {
            throw std::invalid_argument(
                "block_dim * block_dim must be <= 1024");
        }
    }
    catch (const std::exception &error)
    {
        std::cerr << "Argument error: " << error.what() << "\n";
        std::cerr << "Usage: " << argv[0]
                  << " <matrix_size> <block_dim> <iterations>\n";
        return 1;
    }

    size_t elements = static_cast<size_t>(n) * static_cast<size_t>(n);
    size_t bytes = elements * sizeof(float);

    std::vector<float> h_a(elements, 1.0f);
    std::vector<float> h_b(elements, 2.0f);
    std::vector<float> h_b_t(elements, 0.0f);
    std::vector<float> h_c(elements, 0.0f);

    for (int row = 0; row < n; row++)
    {
        for (int col = 0; col < n; col++)
        {
            h_b_t[col * n + row] = h_b[row * n + col];
        }
    }

    float *d_a = nullptr;
    float *d_b = nullptr;
    float *d_b_t = nullptr;
    float *d_c = nullptr;

    CHECK_CUDA(cudaMalloc(&d_a, bytes));
    CHECK_CUDA(cudaMalloc(&d_b, bytes));
    CHECK_CUDA(cudaMalloc(&d_b_t, bytes));
    CHECK_CUDA(cudaMalloc(&d_c, bytes));

    CHECK_CUDA(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b_t, h_b_t.data(), bytes, cudaMemcpyHostToDevice));

    dim3 block(block_dim, block_dim);
    dim3 grid((n + block.x - 1) / block.x,
              (n + block.y - 1) / block.y);

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;

    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    double flops = 2.0 * static_cast<double>(n) *
                   static_cast<double>(n) *
                   static_cast<double>(n);
    double matrix_mib = static_cast<double>(bytes) / (1024.0 * 1024.0);

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "matmul_memory_layout\n";
    std::cout << "N                   = " << n << " x " << n << "\n";
    std::cout << "Matrix bytes        = " << matrix_mib << " MiB each\n";
    std::cout << "Block dim           = " << block.x << " x " << block.y
              << "\n";
    std::cout << "Grid dim            = " << grid.x << " x " << grid.y
              << "\n";
    std::cout << "Warm-up launches    = " << warmups << "\n";
    std::cout << "Iterations          = " << iterations << "\n\n";

    bool all_correct = true;

    for (int variant = 0; variant < 2; variant++)
    {
        bool use_transposed_b = (variant == 1);
        const char *variant_name =
            use_transposed_b ? "transposed_b" : "normal_b";

        for (int i = 0; i < warmups; i++)
        {
            if (use_transposed_b)
            {
                matmul_transposed_b<<<grid, block>>>(d_a, d_b_t, d_c, n);
            }
            else
            {
                matmul_normal_b<<<grid, block>>>(d_a, d_b, d_c, n);
            }

            CHECK_CUDA(cudaGetLastError());
        }

        CHECK_CUDA(cudaDeviceSynchronize());

        std::vector<float> times(iterations);

        for (int i = 0; i < iterations; i++)
        {
            CHECK_CUDA(cudaEventRecord(start));

            if (use_transposed_b)
            {
                matmul_transposed_b<<<grid, block>>>(d_a, d_b_t, d_c, n);
            }
            else
            {
                matmul_normal_b<<<grid, block>>>(d_a, d_b, d_c, n);
            }

            CHECK_CUDA(cudaGetLastError());
            CHECK_CUDA(cudaEventRecord(stop));
            CHECK_CUDA(cudaEventSynchronize(stop));
            CHECK_CUDA(cudaEventElapsedTime(&times[i], start, stop));
        }

        CHECK_CUDA(cudaMemcpy(h_c.data(), d_c, bytes,
                              cudaMemcpyDeviceToHost));

        float sum = 0.0f;

        for (float time : times)
        {
            sum += time;
        }

        float average = sum / static_cast<float>(iterations);
        float minimum = *std::min_element(times.begin(), times.end());
        float maximum = *std::max_element(times.begin(), times.end());
        double gflop_s = flops / (average / 1000.0) / 1e9;
        bool correct = verify_result(h_c, 2.0f * static_cast<float>(n));
        all_correct = all_correct && correct;

        std::cout << "Kernel variant      = " << variant_name << "\n";
        std::cout << "Min kernel time     = " << minimum << " ms\n";
        std::cout << "Max kernel time     = " << maximum << " ms\n";
        std::cout << "Average kernel time = " << average << " ms\n";
        std::cout << "Achieved throughput = " << gflop_s << " GFLOP/s\n";
        std::cout << "Result              = "
                  << (correct ? "PASS" : "FAIL") << "\n\n";
    }

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_b_t));
    CHECK_CUDA(cudaFree(d_c));

    return all_correct ? 0 : 1;
}
