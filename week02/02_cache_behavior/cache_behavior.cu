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
    const char *label;
    int working_set_elements;
};

/*
Learning task for this lab:

You will finish one kernel that repeatedly reads from a bounded working
set. Each thread writes one output value, but it performs many input reads
before writing. By changing only the working-set size, you can observe when
cache reuse is strong and when the workload becomes closer to streaming
from DRAM.

Goal:

- Practice reasoning about L2/cache-friendly reuse.
- Compare small working sets against large working sets.
- Separate "good coalescing" from "good cache reuse".

The benchmark harness below is complete. Your job is to fill in the kernel
TODO without changing the cases, timing code, or verification code.
*/

__global__ void repeated_working_set_read(const float *input,
                                          float *output,
                                          int output_elements,
                                          int working_set_elements,
                                          int repeat_reads)
{
    // Task: implement the repeated working-set read pattern.
    //
    // Each thread should compute one output element. For that output, it
    // repeatedly reads from the active working set and accumulates the
    // values into a register.
    //
    // The working-set size is always a power of two, so wrapping can be
    // done with:
    //
    //     mask = working_set_elements - 1
    //
    // The intended access pattern is:
    //
    //     base        = output_index & mask
    //     input_index = (base + repeat_index * 131) & mask
    //     sum        += input[input_index]
    //     output[output_index] = sum
    //
    // TODO 1: Compute this thread's global output index.
    //
    // TODO 2: Return early if the output index is outside output_elements.
    //
    // TODO 3: Compute the working-set mask and base input index.
    //
    // TODO 4: Loop repeat_reads times, read from input[input_index], and
    // accumulate into a local float.
    //
    // TODO 5: Write the final sum to output[output_index].
    //
    // Observation goal after implementation:
    //
    // Small working sets should have strong cache reuse. As the working set
    // grows, useful bandwidth should eventually drop because more reads
    // have to come from lower levels of the memory hierarchy.
    int output_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (output_index > output_elements)
        return;
    int mask = working_set_elements - 1;
    int base = output_index & mask;
    float sum = 0;
    for (int repeat_index = 0; repeat_index < repeat_reads; ++repeat_index)
    {
        int input_index = (base + repeat_index * 131) & mask;
        sum += input[input_index];
    }
    output[output_index] = sum;
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

float expected_value(const std::vector<float> &input,
                     size_t output_index,
                     int working_set_elements,
                     int repeat_reads)
{
    size_t mask = static_cast<size_t>(working_set_elements - 1);
    size_t base = output_index & mask;
    float sum = 0.0f;

    for (int repeat = 0; repeat < repeat_reads; repeat++)
    {
        size_t input_index =
            (base + static_cast<size_t>(repeat) * 131ULL) & mask;

        sum += input[input_index];
    }

    return sum;
}

bool verify_result(const std::vector<float> &input,
                   const std::vector<float> &output,
                   int working_set_elements,
                   int repeat_reads)
{
    size_t samples = std::min<size_t>(output.size(), 4096);
    size_t step = std::max<size_t>(1, output.size() / samples);

    for (size_t i = 0; i < output.size(); i += step)
    {
        float expected =
            expected_value(input, i, working_set_elements, repeat_reads);

        if (std::fabs(output[i] - expected) > 1e-3f)
        {
            return false;
        }
    }

    return true;
}

int main(int argc, char **argv)
{
    int output_elements = 1 << 20;
    int threads = 256;
    int repeat_reads = 64;
    int iterations = 100;
    const int warmups = 3;

    try
    {
        if (argc > 1)
        {
            output_elements =
                parse_positive_int(argv[1], "output_elements");
        }

        if (argc > 2)
        {
            threads = parse_positive_int(argv[2], "threads_per_block");
        }

        if (argc > 3)
        {
            repeat_reads = parse_positive_int(argv[3], "repeat_reads");
        }

        if (argc > 4)
        {
            iterations = parse_positive_int(argv[4], "iterations");
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
                  << " <output_elements> <threads_per_block>"
                  << " <repeat_reads> <iterations>\n";
        return 1;
    }

    std::vector<Case> cases = {
        {"4 KiB", 1024},
        {"64 KiB", 16384},
        {"1 MiB", 262144},
        {"8 MiB", 2097152},
        {"64 MiB", 16777216},
        {"256 MiB", 67108864},
    };

    int largest_working_set = cases.back().working_set_elements;
    size_t input_elements = static_cast<size_t>(largest_working_set);
    size_t output_count = static_cast<size_t>(output_elements);
    size_t input_bytes = input_elements * sizeof(float);
    size_t output_bytes = output_count * sizeof(float);

    std::vector<float> h_input(input_elements);
    std::vector<float> h_output(output_count, 0.0f);

    for (size_t i = 0; i < h_input.size(); i++)
    {
        h_input[i] = static_cast<float>((i % 251) + 1) * 0.001f;
    }

    float *d_input = nullptr;
    float *d_output = nullptr;

    CHECK_CUDA(cudaMalloc(&d_input, input_bytes));
    CHECK_CUDA(cudaMalloc(&d_output, output_bytes));

    CHECK_CUDA(cudaMemcpy(d_input,
                          h_input.data(),
                          input_bytes,
                          cudaMemcpyHostToDevice));

    int blocks = (output_elements + threads - 1) / threads;
    double input_mib = static_cast<double>(input_bytes) / (1024.0 * 1024.0);
    double output_mib = static_cast<double>(output_bytes) / (1024.0 * 1024.0);

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;

    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "cache_behavior\n";
    std::cout << "Output elements     = " << output_elements << "\n";
    std::cout << "Largest input       = " << input_mib << " MiB\n";
    std::cout << "Output bytes        = " << output_mib << " MiB\n";
    std::cout << "Threads/block       = " << threads << "\n";
    std::cout << "Blocks              = " << blocks << "\n";
    std::cout << "Repeat reads/thread = " << repeat_reads << "\n";
    std::cout << "Warm-up launches    = " << warmups << "\n";
    std::cout << "Iterations          = " << iterations << "\n\n";

    bool all_correct = true;

    for (const Case &benchmark_case : cases)
    {
        CHECK_CUDA(cudaMemset(d_output, 0, output_bytes));

        for (int i = 0; i < warmups; i++)
        {
            repeated_working_set_read<<<blocks, threads>>>(
                d_input,
                d_output,
                output_elements,
                benchmark_case.working_set_elements,
                repeat_reads);

            CHECK_CUDA(cudaGetLastError());
        }

        CHECK_CUDA(cudaDeviceSynchronize());

        std::vector<float> times(iterations);

        for (int i = 0; i < iterations; i++)
        {
            CHECK_CUDA(cudaEventRecord(start));

            repeated_working_set_read<<<blocks, threads>>>(
                d_input,
                d_output,
                output_elements,
                benchmark_case.working_set_elements,
                repeat_reads);

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
        double useful_bytes =
            static_cast<double>(output_bytes) *
            (static_cast<double>(repeat_reads) + 1.0);
        double bandwidth_gb_s = useful_bytes / (average / 1000.0) / 1e9;
        bool correct = verify_result(h_input,
                                     h_output,
                                     benchmark_case.working_set_elements,
                                     repeat_reads);
        all_correct = all_correct && correct;

        std::cout << "Working set         = " << benchmark_case.label
                  << "\n";
        std::cout << "Working-set elements= "
                  << benchmark_case.working_set_elements << "\n";
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
