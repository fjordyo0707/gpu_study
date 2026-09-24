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

enum class WorkloadKind
{
    MemoryStream,
    ComputeChain,
    Mixed,
};

struct Case
{
    const char *name;
    WorkloadKind kind;
    int fma_repeats;
    double bytes_per_element;
    double flops_per_element;
};

/*
Learning task for this lab:

You will implement three kernels that intentionally stress different parts
of the GPU:

- memory_stream_kernel: lots of global memory traffic and little math
- compute_chain_kernel: little memory traffic and lots of dependent math
- mixed_kernel: a middle point with two inputs and repeated arithmetic

Keep the memory traffic described in each kernel comment. If you add extra
loads or stores, the arithmetic-intensity calculation in the harness no
longer matches the code.
*/

__global__ void memory_stream_kernel(const float *a,
                                     const float *b,
                                     const float *c,
                                     float *output,
                                     int n)
{
    // Goal:
    //
    //     output[i] = a[i] * 1.25f + b[i] * 0.75f + c[i]
    //
    // Useful traffic per element:
    //
    //     read a[i], b[i], c[i] = 12 bytes
    //     write output[i]      = 4 bytes
    //     total                = 16 bytes
    //
    // Useful FLOPs per element:
    //
    //     two multiplies + two adds = 4 FLOPs
    //
    // TODO 1: Compute the global index.
    // TODO 2: Guard against index >= n.
    // TODO 3: Load a[i], b[i], and c[i].
    // TODO 4: Compute the expression above.
    // TODO 5: Store the result.
}

__global__ void compute_chain_kernel(const float *a,
                                     float *output,
                                     int n,
                                     int fma_repeats)
{
    // Goal:
    //
    //     x = a[i]
    //     repeat fma_repeats times:
    //         x = fmaf(x, 1.000001f, 0.000001f)
    //     output[i] = x
    //
    // Useful traffic per element:
    //
    //     read a[i]       = 4 bytes
    //     write output[i] = 4 bytes
    //     total           = 8 bytes
    //
    // Useful FLOPs per element:
    //
    //     2 * fma_repeats
    //
    // TODO 1: Compute the global index.
    // TODO 2: Guard against index >= n.
    // TODO 3: Load x = a[i].
    // TODO 4: Run the dependent FMA loop.
    // TODO 5: Store x.
}

__global__ void mixed_kernel(const float *a,
                             const float *b,
                             float *output,
                             int n,
                             int fma_repeats)
{
    // Goal:
    //
    //     x = a[i] + b[i]
    //     repeat fma_repeats times:
    //         x = fmaf(x, 1.000001f, 0.000001f)
    //     output[i] = x
    //
    // Useful traffic per element:
    //
    //     read a[i], b[i] = 8 bytes
    //     write output[i] = 4 bytes
    //     total           = 12 bytes
    //
    // Useful FLOPs per element:
    //
    //     one add + 2 * fma_repeats
    //
    // TODO 1: Compute the global index.
    // TODO 2: Guard against index >= n.
    // TODO 3: Load a[i] and b[i].
    // TODO 4: Add them and run the dependent FMA loop.
    // TODO 5: Store x.
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

float apply_fma_chain(float x, int fma_repeats)
{
    for (int i = 0; i < fma_repeats; i++)
    {
        x = std::fma(x, 1.000001f, 0.000001f);
    }

    return x;
}

float expected_value(const Case &benchmark_case,
                     float a,
                     float b,
                     float c)
{
    switch (benchmark_case.kind)
    {
    case WorkloadKind::MemoryStream:
        return a * 1.25f + b * 0.75f + c;
    case WorkloadKind::ComputeChain:
        return apply_fma_chain(a, benchmark_case.fma_repeats);
    case WorkloadKind::Mixed:
        return apply_fma_chain(a + b, benchmark_case.fma_repeats);
    }

    return 0.0f;
}

bool verify_result(const std::vector<float> &a,
                   const std::vector<float> &b,
                   const std::vector<float> &c,
                   const std::vector<float> &output,
                   const Case &benchmark_case)
{
    size_t samples = std::min<size_t>(output.size(), 4096);
    size_t step = std::max<size_t>(1, output.size() / samples);

    for (size_t i = 0; i < output.size(); i += step)
    {
        float expected = expected_value(benchmark_case, a[i], b[i], c[i]);

        if (std::fabs(output[i] - expected) > 1e-3f)
        {
            return false;
        }
    }

    return true;
}

void launch_case(const Case &benchmark_case,
                 const float *d_a,
                 const float *d_b,
                 const float *d_c,
                 float *d_output,
                 int n,
                 int blocks,
                 int threads)
{
    switch (benchmark_case.kind)
    {
    case WorkloadKind::MemoryStream:
        memory_stream_kernel<<<blocks, threads>>>(d_a, d_b, d_c, d_output, n);
        return;
    case WorkloadKind::ComputeChain:
        compute_chain_kernel<<<blocks, threads>>>(
            d_a, d_output, n, benchmark_case.fma_repeats);
        return;
    case WorkloadKind::Mixed:
        mixed_kernel<<<blocks, threads>>>(
            d_a, d_b, d_output, n, benchmark_case.fma_repeats);
        return;
    }
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
        {"memory_stream", WorkloadKind::MemoryStream, 0, 16.0, 4.0},
        {"compute_chain_64", WorkloadKind::ComputeChain, 64, 8.0, 128.0},
        {"compute_chain_512", WorkloadKind::ComputeChain, 512, 8.0, 1024.0},
        {"mixed_64", WorkloadKind::Mixed, 64, 12.0, 129.0},
    };

    size_t element_count = static_cast<size_t>(n);
    size_t bytes = element_count * sizeof(float);
    double data_mib = static_cast<double>(bytes) / (1024.0 * 1024.0);

    std::vector<float> h_a(element_count);
    std::vector<float> h_b(element_count);
    std::vector<float> h_c(element_count);
    std::vector<float> h_output(element_count, 0.0f);

    for (size_t i = 0; i < element_count; i++)
    {
        h_a[i] = static_cast<float>((i % 997) + 1) * 0.001f;
        h_b[i] = static_cast<float>((i % 389) + 1) * 0.002f;
        h_c[i] = static_cast<float>((i % 127) + 1) * 0.003f;
    }

    float *d_a = nullptr;
    float *d_b = nullptr;
    float *d_c = nullptr;
    float *d_output = nullptr;

    CHECK_CUDA(cudaMalloc(&d_a, bytes));
    CHECK_CUDA(cudaMalloc(&d_b, bytes));
    CHECK_CUDA(cudaMalloc(&d_c, bytes));
    CHECK_CUDA(cudaMalloc(&d_output, bytes));

    CHECK_CUDA(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_c, h_c.data(), bytes, cudaMemcpyHostToDevice));

    int blocks = (n + threads - 1) / threads;

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;

    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "memory_vs_compute\n";
    std::cout << "Elements            = " << n << "\n";
    std::cout << "Array bytes         = " << data_mib << " MiB\n";
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
            launch_case(benchmark_case,
                        d_a,
                        d_b,
                        d_c,
                        d_output,
                        n,
                        blocks,
                        threads);
            CHECK_CUDA(cudaGetLastError());
        }

        CHECK_CUDA(cudaDeviceSynchronize());

        std::vector<float> times(iterations);

        for (int i = 0; i < iterations; i++)
        {
            CHECK_CUDA(cudaEventRecord(start));

            launch_case(benchmark_case,
                        d_a,
                        d_b,
                        d_c,
                        d_output,
                        n,
                        blocks,
                        threads);

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
            static_cast<double>(n) * benchmark_case.bytes_per_element;
        double flops =
            static_cast<double>(n) * benchmark_case.flops_per_element;
        double arithmetic_intensity = flops / useful_bytes;
        double gflops = flops / (average / 1000.0) / 1e9;
        double bandwidth_gb_s = useful_bytes / (average / 1000.0) / 1e9;

        bool correct = verify_result(h_a,
                                     h_b,
                                     h_c,
                                     h_output,
                                     benchmark_case);
        all_correct = all_correct && correct;

        std::cout << "Workload            = " << benchmark_case.name << "\n";
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

    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));
    CHECK_CUDA(cudaFree(d_output));

    return all_correct ? 0 : 1;
}
