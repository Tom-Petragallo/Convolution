CXX=g++
CXXFLAGS=-O3 -march=native
LDLIBS=`pkg-config --libs opencv`


all: main-cu

main: main.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDLIBS)

main-cu: main.cu
	nvcc -o $@ $< $(LDLIBS)

.PHONY: clean

clean:
	rm main-cu main out*