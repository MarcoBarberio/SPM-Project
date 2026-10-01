CXX        = g++
MPICXX     = mpicxx

INCLUDES   = -I./include
CXXFLAGS   = -std=c++20 -Wall -pthread
SEQFLAGS   = -O3
MPIFLAGS   = -O3

TARGET_SEQ = build/hashjoin_seq
TARGET_MPI = build/hashjoin_mpi

SEQ_SRC    = src/hashjoin_seq.cpp
MPI_SRC    = src/hashjoin_mpi.cpp

COMMON_HEADERS = include/utilities.hpp

.PHONY: all clean

all: build $(TARGET_SEQ) $(TARGET_MPI)

build:
	mkdir -p build

$(TARGET_SEQ): $(SEQ_SRC) $(COMMON_HEADERS) | build
	$(CXX) $(INCLUDES) $(CXXFLAGS) $(SEQFLAGS) $(SEQ_SRC) -o $@

$(TARGET_MPI): $(MPI_SRC) $(COMMON_HEADERS) | build
	$(MPICXX) $(INCLUDES) $(CXXFLAGS) $(MPIFLAGS) $(MPI_SRC) -o $@

clean:
	-rm -f $(TARGET_SEQ) $(TARGET_MPI)