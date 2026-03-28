#ifndef MAPPING_CUDA_HPP
#define MAPPING_CUDA_HPP

#include <cstdint>
#include <vector>

void generate_mapping_cuda(const std::vector<uint64_t>& data, std::vector<uint64_t>& mapping, size_t n, int k,
                           uint64_t a, uint64_t b, double* time_h2d, double* time_kernel, double* time_d2h);
#endif