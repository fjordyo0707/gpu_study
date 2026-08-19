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
    const char *pattern;
    int parameter;
};

__global__ void copy_offset(const float *input,
                            float *output,
                            int n,
                            int offset)
{
    // TODO: Compute this thread's global element index.
    //
    // TODO: If the element is in range, copy from the offset input
    // address into the matching output element.
    //
    // The access pattern should be:
    //
    //     output[i] = input[i + offset]
    //
    // Think about what addresses neighboring threads in a warp read.
    (void)input;
    (void)output;
    (void)n;
    (void)offset;
}

__global__ void copy_stride(const float *input,
                            float *output,
                            int n,
                            int stride)
{
    // TODO: Compute this thread's global element index.
    //
    // TODO: If the element is in range, copy from the strided input
    // address into the matching output element.
    //
    // The access pattern should be:
    //
    //     output[i] = input[i * stride]
    //
    // Use a wide enough integer type for the input index.
    (void)input;
    (void)output;
    (void)n;
    (void)stride;
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
                   const Case &benchmark_case)
{
    for (size_t i = 0; i < output.size(); i++)
    {
        size_t input_index = 0;

        if (std::string(benchmark_case.pattern) == "offset")
        {
            input_index = i + static_cast<size_t>(benchmark_case.parameter);
        }
        else
        {
            input_index = i * static_cast<size_t>(benchmark_case.parameter);
        }

        if (std::fabs(output[i] - input[input_index]) > 1e-6f)
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
        {"offset", 0},
        {"offset", 1},
        {"offset", 2},
        {"offset", 4},
        {"offset", 8},
        {"stride", 2},
        {"stride", 4},
        {"stride", 8},
    };

    int max_offset = 8;
    int max_stride = 8;

    size_t input_elements =
        (static_cast<size_t>(n) - 1) * static_cast<size_t>(max_stride) +
        static_cast<size_t>(max_offset) + 1;
    size_t input_bytes = input_elements * sizeof(float);
    size_t output_elements = static_cast<size_t>(n);
    size_t output_bytes = output_elements * sizeof(float);

    std::vector<float> h_input(input_elements);
    std::vector<float> h_output(output_elements, 0.0f);

    for (size_t i = 0; i < h_input.size(); i++)
    {
        h_input[i] = static_cast<float>((i % 8191) + 1);
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
    double useful_bytes = 2.0 * static_cast<double>(output_bytes);
    double input_mib = static_cast<double>(input_bytes) / (1024.0 * 1024.0);
    double output_mib = static_cast<double>(output_bytes) / (1024.0 * 1024.0);

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;

    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "memory_coalescing\n";
    std::cout << "Elements            = " << n << "\n";
    std::cout << "Input bytes         = " << input_mib << " MiB\n";
    std::cout << "Output bytes        = " << output_mib << " MiB\n";
    std::cout << "Threads/block       = " << threads << "\n";
    std::cout << "Blocks              = " << blocks << "\n";
    std::cout << "Warm-up launches    = " << warmups << "\n";
    std::cout << "Iterations          = " << iterations << "\n\n";

    bool all_correct = true;

    for (const Case &benchmark_case : cases)
    {
        CHECK_CUDA(cudaMemset(d_output, 0, output_bytes));

        for (int i = 0; i < warmups; i++)
        {
            if (std::string(benchmark_case.pattern) == "offset")
            {
                copy_offset<<<blocks, threads>>>(
                    d_input, d_output, n, benchmark_case.parameter);
            }
            else
            {
                copy_stride<<<blocks, threads>>>(
                    d_input, d_output, n, benchmark_case.parameter);
            }

            CHECK_CUDA(cudaGetLastError());
        }

        CHECK_CUDA(cudaDeviceSynchronize());

        std::vector<float> times(iterations);

        for (int i = 0; i < iterations; i++)
        {
            CHECK_CUDA(cudaEventRecord(start));

            if (std::string(benchmark_case.pattern) == "offset")
            {
                copy_offset<<<blocks, threads>>>(
                    d_input, d_output, n, benchmark_case.parameter);
            }
            else
            {
                copy_stride<<<blocks, threads>>>(
                    d_input, d_output, n, benchmark_case.parameter);
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
        double bandwidth_gb_s = useful_bytes / (average / 1000.0) / 1e9;
        bool correct = verify_result(h_input, h_output, benchmark_case);
        all_correct = all_correct && correct;

        std::cout << "Pattern             = " << benchmark_case.pattern
                  << "\n";
        std::cout << "Parameter           = " << benchmark_case.parameter
                  << "\n";
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
