CXX        = g++
NVCC       = nvcc

INCLUDES   = -I./include
CXXFLAGS   = -std=c++20 -Wall -pthread

NOVECFLAGS = -O3 -fno-tree-vectorize
VECFLAGS   = -O3 -march=native -mavx2 -mprefer-vector-width=256 -fopt-info-vec
AVXFLAGS   = -O3 -mavx2 -march=native
CUDAFLAGS  = -O3
PARFLAGS   = -O3

# ============================================================
# TARGETS
# ============================================================

# Modulo 1
TARGET_M1_NOVEC = build/main_baseline
TARGET_M1_VEC   = build/main_autovec
TARGET_M1_AVX   = build/main_avx
TARGET_M1_CUDA  = build/main_cuda

# Modulo 2
TARGET_M2_SEQ   = build/hashjoin_seq
TARGET_M2_PAR   = build/hashjoin_par

# ============================================================
# SOURCES
# ============================================================

# Modulo 1
M1_NOVEC_SRC = src/key_mapping/main_baseline.cpp src/key_mapping/mapping_baseline.cpp
M1_VEC_SRC   = src/key_mapping/main_baseline.cpp src/key_mapping/mapping_baseline.cpp
M1_AVX_SRC   = src/key_mapping/main_avx.cpp src/key_mapping/mapping_avx.cpp
M1_CUDA_SRC  = src/key_mapping/main_cuda.cpp src/key_mapping/mapping_cuda.cu

# Modulo 2
M2_SEQ_SRC   = src/hashjoin_seq.cpp
M2_PAR_SRC   = src/hashjoin_par.cpp

# ============================================================
# HEADERS
# ============================================================

COMMON_HEADERS = include/utilities.hpp
M1_HEADERS     = include/mapping_baseline.hpp
AVX_HEADERS    = include/mapping_avx.hpp
CUDA_HEADERS   = include/mapping_cuda.hpp

.PHONY: all modulo1 modulo2 clean

all: build modulo1 modulo2

modulo1: $(TARGET_M1_NOVEC) $(TARGET_M1_VEC) $(TARGET_M1_AVX) $(TARGET_M1_CUDA)
modulo2: $(TARGET_M2_SEQ) $(TARGET_M2_PAR)

build:
	mkdir -p build

# ============================================================
# MODULO 1
# ============================================================

$(TARGET_M1_NOVEC): $(M1_NOVEC_SRC) $(COMMON_HEADERS) $(M1_HEADERS)
	$(CXX) $(INCLUDES) $(CXXFLAGS) $(NOVECFLAGS) $(M1_NOVEC_SRC) -o $@

$(TARGET_M1_VEC): $(M1_VEC_SRC) $(COMMON_HEADERS) $(M1_HEADERS)
	$(CXX) $(INCLUDES) $(CXXFLAGS) $(VECFLAGS) $(M1_VEC_SRC) -o $@

$(TARGET_M1_AVX): $(M1_AVX_SRC) $(COMMON_HEADERS) $(AVX_HEADERS)
	$(CXX) $(INCLUDES) $(CXXFLAGS) $(AVXFLAGS) $(M1_AVX_SRC) -o $@

$(TARGET_M1_CUDA): $(M1_CUDA_SRC) $(COMMON_HEADERS) $(CUDA_HEADERS)
	$(NVCC) $(INCLUDES) $(CUDAFLAGS) $(M1_CUDA_SRC) -o $@

# ============================================================
# MODULO 2
# ============================================================

$(TARGET_M2_SEQ): $(M2_SEQ_SRC) $(COMMON_HEADERS)
	$(CXX) $(INCLUDES) $(CXXFLAGS) $(PARFLAGS) $(M2_SEQ_SRC) -o $@

$(TARGET_M2_PAR): $(M2_PAR_SRC) $(COMMON_HEADERS)
	$(CXX) $(INCLUDES) $(CXXFLAGS) $(PARFLAGS) $(M2_PAR_SRC) -o $@

clean:
	-rm -f build/*