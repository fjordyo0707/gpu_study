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

__global__ void matmul_tiled_sweep(const float *a,
                                   const float *b,
                                   float *c,
                                   int n)
{
    extern __shared__ float shared[];

    int tile_dim = blockDim.x;
    float *a_tile = shared;
    float *b_tile = shared + tile_dim * tile_dim;

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int local_index = threadIdx.y * tile_dim + threadIdx.x;

    float sum = 0.0f;

    for (int tile_start = 0; tile_start < n; tile_start += tile_dim)
    {
        int a_col = tile_start + threadIdx.x;
        int b_row = tile_start + threadIdx.y;

        a_tile[local_index] =
            (row < n && a_col < n) ? a[row * n + a_col] : 0.0f;

        b_tile[local_index] =
            (b_row < n && col < n) ? b[b_row * n + col] : 0.0f;

        __syncthreads();

        for (int k = 0; k < tile_dim; k++)
        {
            sum += a_tile[threadIdx.y * tile_dim + k] *
                   b_tile[k * tile_dim + threadIdx.x];
        }

        __syncthreads();
    }

    if (row < n && col < n)
    {
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

void validate_tile_dim(int tile_dim)
{
    if (tile_dim > 1024 / tile_dim)
    {
        throw std::invalid_argument("tile_dim * tile_dim must be <= 1024");
    }
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
    int iterations = 20;
    const int warmups = 3;
    std::vector<int> tile_dims = {8, 16, 32};

    try
    {
        if (argc > 1)
        {
            n = parse_positive_int(argv[1], "matrix_size");
        }

        if (argc > 2)
        {
            iterations = parse_positive_int(argv[2], "iterations");
        }

        if (argc > 3)
        {
            tile_dims.clear();

            for (int i = 3; i < argc; i++)
            {
                tile_dims.push_back(parse_positive_int(argv[i], "tile_dim"));
            }
        }

        for (int tile_dim : tile_dims)
        {
            validate_tile_dim(tile_dim);
        }
    }
    catch (const std::exception &error)
    {
        std::cerr << "Argument error: " << error.what() << "\n";
        std::cerr << "Usage: " << argv[0]
                  << " <matrix_size> <iterations> [tile_dim...]\n";
        return 1;
    }

    size_t elements = static_cast<size_t>(n) * static_cast<size_t>(n);
    size_t bytes = elements * sizeof(float);

    std::vector<float> h_a(elements, 1.0f);
    std::vector<float> h_b(elements, 2.0f);
    std::vector<float> h_c(elements, 0.0f);

    float *d_a = nullptr;
    float *d_b = nullptr;
    float *d_c = nullptr;

    CHECK_CUDA(cudaMalloc(&d_a, bytes));
    CHECK_CUDA(cudaMalloc(&d_b, bytes));
    CHECK_CUDA(cudaMalloc(&d_c, bytes));

    CHECK_CUDA(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;

    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    double flops = 2.0 * static_cast<double>(n) *
                   static_cast<double>(n) *
                   static_cast<double>(n);
    double matrix_mib = static_cast<double>(bytes) / (1024.0 * 1024.0);

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "matmul_tile_sweep\n";
    std::cout << "N                   = " << n << " x " << n << "\n";
    std::cout << "Matrix bytes        = " << matrix_mib << " MiB each\n";
    std::cout << "Warm-up launches    = " << warmups << "\n";
    std::cout << "Iterations          = " << iterations << "\n\n";

    bool all_correct = true;

    for (int tile_dim : tile_dims)
    {
        dim3 block(tile_dim, tile_dim);
        dim3 grid((n + block.x - 1) / block.x,
                  (n + block.y - 1) / block.y);
        size_t shared_bytes = 2 * static_cast<size_t>(tile_dim) *
                              static_cast<size_t>(tile_dim) * sizeof(float);

        for (int i = 0; i < warmups; i++)
        {
            matmul_tiled_sweep<<<grid, block, shared_bytes>>>(
                d_a, d_b, d_c, n);
            CHECK_CUDA(cudaGetLastError());
        }

        CHECK_CUDA(cudaDeviceSynchronize());

        std::vector<float> times(iterations);

        for (int i = 0; i < iterations; i++)
        {
            CHECK_CUDA(cudaEventRecord(start));

            matmul_tiled_sweep<<<grid, block, shared_bytes>>>(
                d_a, d_b, d_c, n);
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
        double shared_kib = static_cast<double>(shared_bytes) / 1024.0;
        bool correct = verify_result(h_c, 2.0f * static_cast<float>(n));
        all_correct = all_correct && correct;

        std::cout << "Tile dim            = " << tile_dim << "\n";
        std::cout << "Threads/block       = " << block.x * block.y << "\n";
        std::cout << "Grid dim            = " << grid.x << " x " << grid.y
                  << "\n";
        std::cout << "Shared memory/block = " << shared_kib << " KiB\n";
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
    CHECK_CUDA(cudaFree(d_c));

    return all_correct ? 0 : 1;
}
