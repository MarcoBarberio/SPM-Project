CXX      = g++
INCLUDES = -I./include
CXXFLAGS = -std=c++20 -Wall -pthread -O3

TARGET_SEQ = build/hashjoin_seq
TARGET_PAR = build/hashjoin_par

SRC_SEQ = src/hashjoin_seq.cpp
SRC_PAR = src/hashjoin_par.cpp

HEADERS = include/utilities.hpp

.PHONY: all clean

all: $(TARGET_SEQ) $(TARGET_PAR)

build:
	mkdir -p build

$(TARGET_SEQ): $(SRC_SEQ) $(HEADERS) | build
	$(CXX) $(INCLUDES) $(CXXFLAGS) $(SRC_SEQ) -o $@

$(TARGET_PAR): $(SRC_PAR) $(HEADERS) | build
	$(CXX) $(INCLUDES) $(CXXFLAGS) $(SRC_PAR) -o $@

clean:
	rm -f build/*