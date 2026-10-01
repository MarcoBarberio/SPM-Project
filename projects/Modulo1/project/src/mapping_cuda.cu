
#include "hpc_helpers.hpp"
#include <cstdint>
#include <vector>
/** Computes the hash for each element in the input vector using the multiply-shift-add hash function. */
__global__ void mapping_kernel(const uint64_t* data, uint64_t* mapping, size_t n, uint64_t a, uint64_t b, int shift)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        mapping[i] = (a * data[i] + b) >> shift;
    }
}
/** Generates a hash mapping for the given keys using the multiply-shift-add hash function.
 * @param data The input vector containing the keys to be hashed.
 * @param mapping The output vector where the resulting hash values will be stored.
 * @param n The number of elements in the input and output vectors.
 * @param k The number of bits to use for the hash output (the output will be in the range [0, 2^k - 1]).
 * @param a The multiplier for the hash function.
 * @param b The addend for the hash function.
 * @param time_h2d (optional) The time taken to transfer data from host to device (in seconds).
 * @param time_kernel (optional) The time taken to execute the kernel on the device (in seconds).
 * @param time_d2h (optional) The time taken to transfer data from device to host (in seconds).
 * */

void generate_mapping_cuda(const std::vector<uint64_t>& data, std::vector<uint64_t>& mapping, size_t n, int k,
                           uint64_t a, uint64_t b, double* time_h2d, double* time_kernel, double* time_d2h)
{
    const int shift = 64 - k;

    uint64_t *d_data, *d_mapping;

    // Allocate device memory
    cudaMalloc(&d_data, n * sizeof(uint64_t));
    cudaMalloc(&d_mapping, n * sizeof(uint64_t));

    TIMERSTART(h2d);
    // Copy data from host to device
    cudaMemcpy(d_data, data.data(), n * sizeof(uint64_t), H2D);
    cudaDeviceSynchronize();
    TIMERSTOP(h2d);

    // Calculate block and grid sizes
    int blockSize = 256;
    int numBlocks = SDIV(n, blockSize);

    TIMERSTART(kernel);
    // Launch kernel
    mapping_kernel<<<numBlocks, blockSize>>>(d_data, d_mapping, n, a, b, shift);
    cudaDeviceSynchronize();
    TIMERSTOP(kernel);

    TIMERSTART(d2h);
    // Copy results from device to host
    cudaMemcpy(mapping.data(), d_mapping, n * sizeof(uint64_t), D2H);
    cudaDeviceSynchronize();
    TIMERSTOP(d2h);

    cudaFree(d_data);
    cudaFree(d_mapping);

    // Convert times to seconds and store in output parameters
    if (time_h2d)
        *time_h2d = timeh2d / 1000.0;
    if (time_kernel)
        *time_kernel = timekernel / 1000.0;
    if (time_d2h)
        *time_d2h = timed2h / 1000.0;
}