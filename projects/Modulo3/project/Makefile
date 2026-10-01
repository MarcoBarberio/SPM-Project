CXX      = g++
INCLUDES = -I./include

CXXFLAGS = -std=c++20 -Wall -pthread
SEQFLAGS = -O3
OMPFLAGS = -O3 -fopenmp

# ============================================================
# TARGETS
# ============================================================

TARGET_SEQ  = build/hashjoin_seq
TARGET_FOR  = build/hashjoin_openmp_for
TARGET_TASK = build/hashjoin_openmp_task

# ============================================================
# SOURCES
# ============================================================

SEQ_SRC  = src/hashjoin_seq.cpp
FOR_SRC  = src/hashjoin_openmp_for.cpp
TASK_SRC = src/hashjoin_openmp_task.cpp

# ============================================================
# HEADERS
# ============================================================

COMMON_HEADERS = include/utilities.hpp

# ============================================================
# RULES
# ============================================================

.PHONY: all modulo3 clean

# Default target: running just "make" compiles Module 3
all: modulo3

modulo3: build $(TARGET_SEQ) $(TARGET_FOR) $(TARGET_TASK)

build:
	mkdir -p build

$(TARGET_SEQ): $(SEQ_SRC) $(COMMON_HEADERS) | build
	$(CXX) $(INCLUDES) $(CXXFLAGS) $(SEQFLAGS) $(SEQ_SRC) -o $@

$(TARGET_FOR): $(FOR_SRC) $(COMMON_HEADERS) | build
	$(CXX) $(INCLUDES) $(CXXFLAGS) $(OMPFLAGS) $(FOR_SRC) -o $@

$(TARGET_TASK): $(TASK_SRC) $(COMMON_HEADERS) | build
	$(CXX) $(INCLUDES) $(CXXFLAGS) $(OMPFLAGS) $(TASK_SRC) -o $@

clean:
	-rm -f build/*