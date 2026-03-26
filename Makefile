CXX        = g++ -std=c++20

INCLUDES   = -I./include
CXXFLAGS  += -Wall

NOVECFLAGS = -O3 -fno-tree-vectorize
VECFLAGS   = -O3 -march=native -mavx2 -mprefer-vector-width=256  -fopt-info-vec
AVXFLAGS   = -O3 -mavx2 -march=native

TARGET_NOVEC = build/main_baseline
TARGET_VEC   = build/main_autovec
TARGET_AVX   = build/main_avx

.PHONY: all clean

all: build $(TARGET_NOVEC) $(TARGET_VEC) $(TARGET_AVX)

build:
	mkdir -p build

$(TARGET_NOVEC):
	$(CXX) $(INCLUDES) $(CXXFLAGS) $(NOVECFLAGS) \
	src/main_baseline.cpp src/mapping_baseline.cpp \
	-o $@

$(TARGET_VEC):
	$(CXX) $(INCLUDES) $(CXXFLAGS) $(VECFLAGS) \
	src/main_baseline.cpp src/mapping_baseline.cpp \
	-o $@

$(TARGET_AVX):
	$(CXX) $(INCLUDES) $(CXXFLAGS) $(AVXFLAGS) \
	src/main_avx.cpp src/mapping_avx.cpp \
	-o $@

clean:
	-rm -f build/*