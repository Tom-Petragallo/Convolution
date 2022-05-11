CXX=g++
CXXFLAGS=-O3 -march=native
LDLIBS1=-lm -lIL
LDLIBS2=`pkg-config --libs opencv`


all: main-cu

main-cu: main.cu
	nvcc -o $@ $< $(LDLIBS2)

.PHONY: clean

clean:
	rm main-cu