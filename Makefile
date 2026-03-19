CXX        = g++ -std=c++20

INCLUDES   = -I./include
CXXFLAGS  += -Wall

NOVECFLAGS = -O3 -fno-tree-vectorize
VECFLAGS   = -O3 -march=native -ffast-math -fopt-info-vec

SOURCES    = $(wildcard src/*.cpp)

TARGET_NOVEC = build/main_novec
TARGET_VEC   = build/main_vec

.PHONY: all clean

all: build $(TARGET_NOVEC) $(TARGET_VEC)


$(TARGET_NOVEC): $(SOURCES)
	$(CXX) $(INCLUDES) $(CXXFLAGS) $(NOVECFLAGS) -o $@ $(SOURCES)

$(TARGET_VEC): $(SOURCES)
	$(CXX) $(INCLUDES) $(CXXFLAGS) $(VECFLAGS) -o $@ $(SOURCES)

clean:
	-rm -f build/*