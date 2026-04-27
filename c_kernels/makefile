CXX = g++
CXXFLAGS = -std=c++17 -fopenmp -O2 -fPIC -shared
PYTHON ?= python
JAXLIB_INCLUDE := $(shell $(PYTHON) -c "import pathlib, jaxlib; print(pathlib.Path(jaxlib.__file__).resolve().parent / 'include')")
INCLUDES = -I$(JAXLIB_INCLUDE)

TARGET = distribution.so
SRCS = distribution.cpp

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $^ -o $@

clean:
	rm -f $(TARGET)
