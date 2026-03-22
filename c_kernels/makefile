CXX = g++
CXXFLAGS = -std=c++17 -fopenmp -O2 -fPIC -shared
# Ensure that the path to jaxlib is correct for your environment
INCLUDES = -I/home/Owner/.local/lib/python3.10/site-packages/jaxlib/include

TARGET = distribution.so
SRCS = distribution.cpp

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $^ -o $@

clean:
	rm -f $(TARGET)
