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

struct Case
{
    int stride;
    int expected_conflict;
};

/*
Learning task for this lab:

You will finish one kernel that stages one value per thread in shared
memory, then repeatedly reads it. The arithmetic is intentionally tiny.
The experiment is about the shared-memory address pattern used by a warp.

Goal:

- Practice using dynamically allocated shared memory.
- Understand why shared memory is split into banks.
- Measure how power-of-two strides create bank conflicts.

The benchmark harness below is complete. Your job is to fill in the kernel
TODO without changing the cases, timing code, or verification code.
*/

__global__ void shared_stride_read(const float *input,
                                   float *output,
                                   int n,
                                   int repeat_accesses,
                                   int stride)
{
    extern __shared__ float shared_values[];

    // Task: implement the shared-memory stride pattern.
    //
    // Each thread should copy one value from global memory into shared
    // memory, synchronize, repeatedly read that shared value, and write the
    // accumulated sum back to global memory.
    //
    // The important line for this experiment is:
    //
    //     shared_index = threadIdx.x * stride
    //
    // For float data, shared-memory bank selection is roughly:
    //
    //     bank = shared_index % 32
    //
    // So stride 1 spreads a warp across banks, while stride 32 sends every
    // lane in a warp to the same bank.
    //
    // TODO 1: Compute this thread's global output index.
    //
    // TODO 2: Compute whether this thread is in range. Do not return before
    // __syncthreads(), because all threads in the block must reach the
    // synchronization point.
    //
    // TODO 3: Compute shared_index = threadIdx.x * stride.
    //
    // TODO 4: Store input[global_index] into shared_values[shared_index]
    // for in-range threads. Store 0.0f for out-of-range threads.
    //
    // TODO 5: Synchronize the block.
    //
    // TODO 6: Repeatedly read shared_values[shared_index] into an
    // accumulator. Use a volatile shared-memory pointer if the compiler
    // tries to optimize the repeated reads away.
    //
    // TODO 7: Write the accumulated result to output[global_index] for
    // in-range threads.
    //
    // Observation goal after implementation:
    //
    // Runtime should generally increase as the expected bank-conflict
    // degree increases.
    (void)input;
    (void)output;
    (void)n;
    (void)repeat_accesses;
    (void)stride;
    (void)shared_values;
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

bool verify_result(const std::vector<float> &input,
                   const std::vector<float> &output,
                   int repeat_accesses)
{
    size_t samples = std::min<size_t>(output.size(), 4096);
    size_t step = std::max<size_t>(1, output.size() / samples);

    for (size_t i = 0; i < output.size(); i += step)
    {
        float expected = input[i] * static_cast<float>(repeat_accesses);

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
    int repeat_accesses = 256;
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
            repeat_accesses = parse_positive_int(argv[3], "repeat_accesses");
        }

        if (argc > 4)
        {
            iterations = parse_positive_int(argv[4], "iterations");
        }

        if (threads > 256)
        {
            throw std::invalid_argument(
                "threads_per_block must be <= 256 for this stride sweep");
        }
    }
    catch (const std::exception &error)
    {
        std::cerr << "Argument error: " << error.what() << "\n";
        std::cerr << "Usage: " << argv[0]
                  << " <elements> <threads_per_block>"
                  << " <repeat_accesses> <iterations>\n";
        return 1;
    }

    std::vector<Case> cases = {
        {1, 1},
        {2, 2},
        {4, 4},
        {8, 8},
        {16, 16},
        {32, 32},
    };

    size_t element_count = static_cast<size_t>(n);
    size_t bytes = element_count * sizeof(float);

    std::vector<float> h_input(element_count);
    std::vector<float> h_output(element_count, 0.0f);

    for (size_t i = 0; i < h_input.size(); i++)
    {
        h_input[i] = static_cast<float>((i % 1021) + 1) * 0.001f;
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

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;

    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "shared_bank_conflicts\n";
    std::cout << "Elements            = " << n << "\n";
    std::cout << "Input bytes         = " << data_mib << " MiB\n";
    std::cout << "Output bytes        = " << data_mib << " MiB\n";
    std::cout << "Threads/block       = " << threads << "\n";
    std::cout << "Blocks              = " << blocks << "\n";
    std::cout << "Repeat accesses     = " << repeat_accesses << "\n";
    std::cout << "Warm-up launches    = " << warmups << "\n";
    std::cout << "Iterations          = " << iterations << "\n\n";

    bool all_correct = true;

    for (const Case &benchmark_case : cases)
    {
        size_t shared_elements =
            static_cast<size_t>(threads - 1) *
            static_cast<size_t>(benchmark_case.stride) +
            1;
        size_t shared_bytes = shared_elements * sizeof(float);
        double shared_kib =
            static_cast<double>(shared_bytes) / 1024.0;

        CHECK_CUDA(cudaMemset(d_output, 0, bytes));

        for (int i = 0; i < warmups; i++)
        {
            shared_stride_read<<<blocks, threads, shared_bytes>>>(
                d_input,
                d_output,
                n,
                repeat_accesses,
                benchmark_case.stride);

            CHECK_CUDA(cudaGetLastError());
        }

        CHECK_CUDA(cudaDeviceSynchronize());

        std::vector<float> times(iterations);

        for (int i = 0; i < iterations; i++)
        {
            CHECK_CUDA(cudaEventRecord(start));

            shared_stride_read<<<blocks, threads, shared_bytes>>>(
                d_input,
                d_output,
                n,
                repeat_accesses,
                benchmark_case.stride);

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
        double useful_bytes =
            static_cast<double>(bytes) *
            (static_cast<double>(repeat_accesses) + 2.0);
        double bandwidth_gb_s = useful_bytes / (average / 1000.0) / 1e9;
        bool correct = verify_result(h_input, h_output, repeat_accesses);
        all_correct = all_correct && correct;

        std::cout << "Stride              = " << benchmark_case.stride
                  << "\n";
        std::cout << "Expected conflict   = "
                  << benchmark_case.expected_conflict << "-way\n";
        std::cout << "Shared memory/block = " << shared_kib << " KiB\n";
        std::cout << "Min kernel time     = " << minimum << " ms\n";
        std::cout << "Max kernel time     = " << maximum << " ms\n";
        std::cout << "Average kernel time = " << average << " ms\n";
        std::cout << "Useful bandwidth    = " << bandwidth_gb_s
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
