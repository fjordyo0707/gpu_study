#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#define CHECK_CUDA(call)                                                 \
    do                                                                   \
    {                                                                    \
        cudaError_t status = (call);                                     \
        if (status != cudaSuccess)                                       \
        {                                                                \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << ": " << cudaGetErrorString(status) << "\n";     \
            return 1;                                                    \
        }                                                                \
    } while (0)

struct Case
{
    const char *name;
    int use_shared;
    int padding;
};

/*
Learning task for this lab:

You will implement two versions of the same small stencil-like workload.
The direct version repeatedly reads the neighborhood from global memory.
The shared version stages the block's neighborhood in shared memory first.

Goal:

- Compare hardware-managed global-memory cache reuse with explicit shared
  memory staging.
- Practice loading a shared-memory tile with a halo region.
- Learn that shared memory helps only when reuse is high enough to repay
  the extra load and synchronization overhead.

The benchmark harness below is complete. Your job is to fill in the two
kernel TODO sections without changing the cases, timing code, or
verification code.
*/

__global__ void stencil_global_direct(const float *input,
                                      float *output,
                                      int n,
                                      int radius,
                                      int repeat_accesses)
{
    // Task: implement the direct global-memory version.
    //
    // Each thread computes one output element:
    //
    //     for repeat in repeat_accesses:
    //         for offset in 0..radius:
    //             sum += input[global_index + offset]
    //     output[global_index] = sum
    //
    // TODO 1: Compute this thread's global output index.
    //
    // TODO 2: Return early if the index is outside n.
    //
    // TODO 3: Accumulate the repeated neighborhood reads from global
    // memory.
    //
    // TODO 4: Write the final sum to output.
    //
    // Observation goal after implementation:
    //
    // This version has simple coalesced global reads. It may already be
    // fast when cache reuse is strong.
    int g_idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (g_idx >= n)
        return;

    float sum = 0.0f;
    for (int i = 0; i < repeat_accesses; ++i)
    {
        for (int j = 0; j <= radius; ++j)
        {
            if ((g_idx + j) < n)
                sum += input[g_idx + j];
        }
    }
    output[g_idx] = sum;
}

__global__ void stencil_shared_tiled(const float *input,
                                     float *output,
                                     int n,
                                     int radius,
                                     int repeat_accesses,
                                     int padding)
{
    extern __shared__ float tile[];

    // Task: implement the shared-memory staged version.
    //
    // The block should stage:
    //
    //     blockDim.x + radius + padding
    //
    // floats into shared memory. The extra radius elements are the halo
    // needed by the last threads in the block. The padding element is an
    // unused tail slot for layout experiments.
    //
    // Use this layout:
    //
    //     tile[threadIdx.x] = input[global_index]
    //
    // Then have the first radius threads load the halo:
    //
    //     tile[blockDim.x + threadIdx.x] =
    //         input[block_start + blockDim.x + threadIdx.x]
    //
    // After synchronizing, each thread should compute the same mathematical
    // result as stencil_global_direct, but read from tile[threadIdx.x +
    // offset] instead of input[global_index + offset].
    //
    // TODO 1: Compute global_index and block_start.
    //
    // TODO 2: Store each in-range thread's main input value into tile.
    //
    // TODO 3: Load the halo elements using the first radius threads. If a
    // halo index reaches beyond n + radius, store 0.0f.
    //
    // TODO 4: Synchronize the block.
    //
    // TODO 5: Accumulate repeated neighborhood reads from shared memory.
    //
    // TODO 6: Write the final sum to output for in-range threads.
    //
    // Important: do not return before __syncthreads(), because every thread
    // in the block must reach the synchronization point.
    int g_idx = blockDim.x * blockIdx.x + threadIdx.x;
    int b_start = blockDim.x * blockIdx.x;

    if (g_idx < n)
        tile[threadIdx.x] = input[g_idx];
    if (threadIdx.x <= radius)
    {
        int halo_idx = b_start + blockDim.x + threadIdx.x;
        if (halo_idx < n)
            tile[blockDim.x + threadIdx.x] = input[halo_idx];
        else
            tile[blockDim.x + threadIdx.x] = 0.0;
    }
    __syncthreads();
    float sum = 0;
    for (int i = 0; i < repeat_accesses; ++i)
    {
        for (int j = 0; j <= radius; ++j)
        {
            sum += tile[threadIdx.x + j];
        }
    }
    output[g_idx] = sum;
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

float expected_value(const std::vector<float> &input,
                     size_t output_index,
                     int radius,
                     int repeat_accesses)
{
    float sum = 0.0f;

    for (int repeat = 0; repeat < repeat_accesses; repeat++)
    {
        for (int offset = 0; offset <= radius; offset++)
        {
            sum += input[output_index + static_cast<size_t>(offset)];
        }
    }

    return sum;
}

bool verify_result(const std::vector<float> &input,
                   const std::vector<float> &output,
                   int radius,
                   int repeat_accesses)
{
    size_t samples = std::min<size_t>(output.size(), 4096);
    size_t step = std::max<size_t>(1, output.size() / samples);

    for (size_t i = 0; i < output.size(); i += step)
    {
        float expected = expected_value(input, i, radius, repeat_accesses);

        if (std::fabs(output[i] - expected) > 1e-3f)
        {
            return false;
        }
    }

    return true;
}

int main(int argc, char **argv)
{
    int n = 1 << 24;
    int threads = 256;
    int radius = 4;
    int repeat_accesses = 32;
    int iterations = 100;
    const int warmups = 3;

    try
    {
        if (argc > 1)
        {
            n = parse_positive_int(argv[1], "elements");
        }

        if (argc > 2)
        {
            threads = parse_positive_int(argv[2], "threads_per_block");
        }

        if (argc > 3)
        {
            radius = parse_positive_int(argv[3], "radius");
        }

        if (argc > 4)
        {
            repeat_accesses = parse_positive_int(argv[4], "repeat_accesses");
        }

        if (argc > 5)
        {
            iterations = parse_positive_int(argv[5], "iterations");
        }

        if (threads > 1024)
        {
            throw std::invalid_argument(
                "threads_per_block must be <= 1024");
        }

        if (radius > threads)
        {
            throw std::invalid_argument(
                "radius must be <= threads_per_block");
        }
    }
    catch (const std::exception &error)
    {
        std::cerr << "Argument error: " << error.what() << "\n";
        std::cerr << "Usage: " << argv[0]
                  << " <elements> <threads_per_block>"
                  << " <radius> <repeat_accesses> <iterations>\n";
        return 1;
    }

    std::vector<Case> cases = {
        {"direct_global", 0, 0},
        {"shared_tiled", 1, 0},
        {"shared_padded", 1, 1},
    };

    size_t output_count = static_cast<size_t>(n);
    size_t input_count = output_count + static_cast<size_t>(radius);
    size_t input_bytes = input_count * sizeof(float);
    size_t output_bytes = output_count * sizeof(float);

    std::vector<float> h_input(input_count);
    std::vector<float> h_output(output_count, 0.0f);

    for (size_t i = 0; i < h_input.size(); i++)
    {
        h_input[i] = static_cast<float>((i % 4093) + 1) * 0.001f;
    }

    float *d_input = nullptr;
    float *d_output = nullptr;

    CHECK_CUDA(cudaMalloc(&d_input, input_bytes));
    CHECK_CUDA(cudaMalloc(&d_output, output_bytes));

    CHECK_CUDA(cudaMemcpy(d_input,
                          h_input.data(),
                          input_bytes,
                          cudaMemcpyHostToDevice));

    int blocks = (n + threads - 1) / threads;
    double input_mib =
        static_cast<double>(input_bytes) / (1024.0 * 1024.0);
    double output_mib =
        static_cast<double>(output_bytes) / (1024.0 * 1024.0);

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;

    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "global_vs_shared_reuse\n";
    std::cout << "Elements            = " << n << "\n";
    std::cout << "Input bytes         = " << input_mib << " MiB\n";
    std::cout << "Output bytes        = " << output_mib << " MiB\n";
    std::cout << "Threads/block       = " << threads << "\n";
    std::cout << "Blocks              = " << blocks << "\n";
    std::cout << "Radius              = " << radius << "\n";
    std::cout << "Repeat accesses     = " << repeat_accesses << "\n";
    std::cout << "Warm-up launches    = " << warmups << "\n";
    std::cout << "Iterations          = " << iterations << "\n\n";

    bool all_correct = true;

    for (const Case &benchmark_case : cases)
    {
        size_t shared_elements =
            static_cast<size_t>(threads + radius + benchmark_case.padding);
        size_t shared_bytes =
            benchmark_case.use_shared ? shared_elements * sizeof(float) : 0;
        double shared_kib =
            static_cast<double>(shared_bytes) / 1024.0;

        CHECK_CUDA(cudaMemset(d_output, 0, output_bytes));

        for (int i = 0; i < warmups; i++)
        {
            if (benchmark_case.use_shared)
            {
                stencil_shared_tiled<<<blocks, threads, shared_bytes>>>(
                    d_input,
                    d_output,
                    n,
                    radius,
                    repeat_accesses,
                    benchmark_case.padding);
            }
            else
            {
                stencil_global_direct<<<blocks, threads>>>(
                    d_input,
                    d_output,
                    n,
                    radius,
                    repeat_accesses);
            }

            CHECK_CUDA(cudaGetLastError());
        }

        CHECK_CUDA(cudaDeviceSynchronize());

        std::vector<float> times(iterations);

        for (int i = 0; i < iterations; i++)
        {
            CHECK_CUDA(cudaEventRecord(start));

            if (benchmark_case.use_shared)
            {
                stencil_shared_tiled<<<blocks, threads, shared_bytes>>>(
                    d_input,
                    d_output,
                    n,
                    radius,
                    repeat_accesses,
                    benchmark_case.padding);
            }
            else
            {
                stencil_global_direct<<<blocks, threads>>>(
                    d_input,
                    d_output,
                    n,
                    radius,
                    repeat_accesses);
            }

            CHECK_CUDA(cudaGetLastError());
            CHECK_CUDA(cudaEventRecord(stop));
            CHECK_CUDA(cudaEventSynchronize(stop));
            CHECK_CUDA(cudaEventElapsedTime(&times[i], start, stop));
        }

        CHECK_CUDA(cudaMemcpy(h_output.data(),
                              d_output,
                              output_bytes,
                              cudaMemcpyDeviceToHost));

        float sum = 0.0f;

        for (float time : times)
        {
            sum += time;
        }

        float average = sum / static_cast<float>(iterations);
        float minimum = *std::min_element(times.begin(), times.end());
        float maximum = *std::max_element(times.begin(), times.end());
        double logical_reads =
            static_cast<double>(radius + 1) *
            static_cast<double>(repeat_accesses);
        double logical_bytes =
            static_cast<double>(output_bytes) * (logical_reads + 1.0);
        double bandwidth_gb_s = logical_bytes / (average / 1000.0) / 1e9;
        bool correct = verify_result(h_input,
                                     h_output,
                                     radius,
                                     repeat_accesses);
        all_correct = all_correct && correct;

        std::cout << "Kernel              = " << benchmark_case.name
                  << "\n";
        std::cout << "Padding             = " << benchmark_case.padding
                  << "\n";
        std::cout << "Shared memory/block = " << shared_kib << " KiB\n";
        std::cout << "Min kernel time     = " << minimum << " ms\n";
        std::cout << "Max kernel time     = " << maximum << " ms\n";
        std::cout << "Average kernel time = " << average << " ms\n";
        std::cout << "Logical bandwidth   = " << bandwidth_gb_s
                  << " GB/s\n";
        std::cout << "Result              = "
                  << (correct ? "PASS" : "FAIL") << "\n\n";
    }

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));

    return all_correct ? 0 : 1;
}
