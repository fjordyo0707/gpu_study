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
    int fma_repeats;
};

/*
Learning task for this lab:

You will finish one kernel that keeps memory traffic fixed while increasing
the amount of floating-point work per element.

Goal:

- Measure arithmetic intensity in FLOP/byte.
- See when a kernel transitions away from pure memory-bandwidth behavior.
- Prepare data for a simple Roofline plot.

The benchmark harness below is complete. Your job is to fill in the kernel
TODO without changing the cases, timing code, or verification code.
*/

__global__ void arithmetic_intensity_kernel(const float *input,
                                            float *output,
                                            int n,
                                            int fma_repeats)
{
    // Task: implement the fixed-memory, variable-compute kernel.
    //
    // Each thread should:
    //
    //     1. Load one input value.
    //     2. Apply fma_repeats dependent FMA operations.
    //     3. Write one output value.
    //
    // The dependency is important. If each FMA depends on the previous
    // value, the compiler and GPU cannot treat the work as independent
    // operations that are easy to remove or reorder.
    //
    // Suggested update:
    //
    //     x = fmaf(x, 1.000001f, 0.000001f)
    //
    // TODO 1: Compute this thread's global output index.
    //
    // TODO 2: Return early if the index is outside n.
    //
    // TODO 3: Load x = input[index].
    //
    // TODO 4: Run the dependent FMA loop.
    //
    // TODO 5: Store x to output[index].
    int g_idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (g_idx >= n)
        return;
    float x = input[g_idx];
    for (int i = 0; i < fma_repeats; ++i)
    {
        x = fmaf(x, 1.000001f, 0.000001f);
    }

    output[g_idx] = x;
    return;
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

float expected_value(float input, int fma_repeats)
{
    float x = input;

    for (int i = 0; i < fma_repeats; i++)
    {
        x = std::fma(x, 1.000001f, 0.000001f);
    }

    return x;
}

bool verify_result(const std::vector<float> &input,
                   const std::vector<float> &output,
                   int fma_repeats)
{
    size_t samples = std::min<size_t>(output.size(), 4096);
    size_t step = std::max<size_t>(1, output.size() / samples);

    for (size_t i = 0; i < output.size(); i += step)
    {
        float expected = expected_value(input[i], fma_repeats);

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
            iterations = parse_positive_int(argv[3], "iterations");
        }

        if (threads > 1024)
        {
            throw std::invalid_argument(
                "threads_per_block must be <= 1024");
        }
    }
    catch (const std::exception &error)
    {
        std::cerr << "Argument error: " << error.what() << "\n";
        std::cerr << "Usage: " << argv[0]
                  << " <elements> <threads_per_block> <iterations>\n";
        return 1;
    }

    std::vector<Case> cases = {
        {0},
        {1},
        {4},
        {16},
        {64},
        {256},
        {1024},
    };

    size_t element_count = static_cast<size_t>(n);
    size_t bytes = element_count * sizeof(float);

    std::vector<float> h_input(element_count);
    std::vector<float> h_output(element_count, 0.0f);

    for (size_t i = 0; i < h_input.size(); i++)
    {
        h_input[i] = static_cast<float>((i % 997) + 1) * 0.001f;
    }

    float *d_input = nullptr;
    float *d_output = nullptr;

    CHECK_CUDA(cudaMalloc(&d_input, bytes));
    CHECK_CUDA(cudaMalloc(&d_output, bytes));

    CHECK_CUDA(cudaMemcpy(d_input,
                          h_input.data(),
                          bytes,
                          cudaMemcpyHostToDevice));

    int blocks = (n + threads - 1) / threads;
    double data_mib = static_cast<double>(bytes) / (1024.0 * 1024.0);
    double useful_bytes = 2.0 * static_cast<double>(bytes);

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;

    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "arithmetic_intensity\n";
    std::cout << "Elements            = " << n << "\n";
    std::cout << "Input bytes         = " << data_mib << " MiB\n";
    std::cout << "Output bytes        = " << data_mib << " MiB\n";
    std::cout << "Threads/block       = " << threads << "\n";
    std::cout << "Blocks              = " << blocks << "\n";
    std::cout << "Warm-up launches    = " << warmups << "\n";
    std::cout << "Iterations          = " << iterations << "\n\n";

    bool all_correct = true;

    for (const Case &benchmark_case : cases)
    {
        CHECK_CUDA(cudaMemset(d_output, 0, bytes));

        for (int i = 0; i < warmups; i++)
        {
            arithmetic_intensity_kernel<<<blocks, threads>>>(
                d_input, d_output, n, benchmark_case.fma_repeats);

            CHECK_CUDA(cudaGetLastError());
        }

        CHECK_CUDA(cudaDeviceSynchronize());

        std::vector<float> times(iterations);

        for (int i = 0; i < iterations; i++)
        {
            CHECK_CUDA(cudaEventRecord(start));

            arithmetic_intensity_kernel<<<blocks, threads>>>(
                d_input, d_output, n, benchmark_case.fma_repeats);

            CHECK_CUDA(cudaGetLastError());
            CHECK_CUDA(cudaEventRecord(stop));
            CHECK_CUDA(cudaEventSynchronize(stop));
            CHECK_CUDA(cudaEventElapsedTime(&times[i], start, stop));
        }

        CHECK_CUDA(cudaMemcpy(h_output.data(),
                              d_output,
                              bytes,
                              cudaMemcpyDeviceToHost));

        float sum = 0.0f;

        for (float time : times)
        {
            sum += time;
        }

        float average = sum / static_cast<float>(iterations);
        float minimum = *std::min_element(times.begin(), times.end());
        float maximum = *std::max_element(times.begin(), times.end());
        double flops =
            2.0 * static_cast<double>(benchmark_case.fma_repeats) *
            static_cast<double>(n);
        double arithmetic_intensity =
            flops / useful_bytes;
        double gflops = flops / (average / 1000.0) / 1e9;
        double bandwidth_gb_s = useful_bytes / (average / 1000.0) / 1e9;
        bool correct = verify_result(h_input,
                                     h_output,
                                     benchmark_case.fma_repeats);
        all_correct = all_correct && correct;

        std::cout << "FMA repeats         = "
                  << benchmark_case.fma_repeats << "\n";
        std::cout << "Arithmetic intensity= "
                  << arithmetic_intensity << " FLOP/byte\n";
        std::cout << "Min kernel time     = " << minimum << " ms\n";
        std::cout << "Max kernel time     = " << maximum << " ms\n";
        std::cout << "Average kernel time = " << average << " ms\n";
        std::cout << "Achieved throughput = " << gflops << " GFLOP/s\n";
        std::cout << "Effective bandwidth = " << bandwidth_gb_s
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
