CXX        = g++ -std=c++20
NVCC       = nvcc

INCLUDES   = -I./include
CXXFLAGS  += -Wall

NOVECFLAGS = -O3 -fno-tree-vectorize
VECFLAGS   = -O3 -march=native -mavx2 -mprefer-vector-width=256 -fopt-info-vec
AVXFLAGS   = -O3 -mavx2 -march=native
CUDAFLAGS  = -O3

TARGET_NOVEC = build/main_baseline
TARGET_VEC   = build/main_autovec
TARGET_AVX   = build/main_avx
TARGET_CUDA  = build/main_cuda

.PHONY: all clean

all: build $(TARGET_NOVEC) $(TARGET_VEC) $(TARGET_AVX) $(TARGET_CUDA)

build:
	mkdir -p build

$(TARGET_NOVEC):
	$(CXX) $(INCLUDES) $(CXXFLAGS) $(NOVECFLAGS) \
	src/key_mapping/main_baseline.cpp src/key_mapping/mapping_baseline.cpp \
	-o $@

$(TARGET_VEC):
	$(CXX) $(INCLUDES) $(CXXFLAGS) $(VECFLAGS) \
	src/key_mapping/main_baseline.cpp src/key_mapping/mapping_baseline.cpp \
	-o $@

$(TARGET_AVX):
	$(CXX) $(INCLUDES) $(CXXFLAGS) $(AVXFLAGS) \
	src/key_mapping/main_avx.cpp src/key_mapping/mapping_avx.cpp \
	-o $@

$(TARGET_CUDA):
	$(NVCC) $(INCLUDES) $(CUDAFLAGS) \
	src/key_mapping/main_cuda.cpp src/key_mapping/mapping_cuda.cu \
	-o $@

clean:
	-rm -f build/*