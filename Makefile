CC=g++-8
INCLUDES=-I${CONDA_PREFIX}/include
FLAGS=-std=c++17 -ffast-math -O3 -mavx2 -lcblas -Wl,-v
DEBUG=-fopt-info-vec-missed 

run: main
	./main

main: main.cpp
	$(CC) -o main main.cpp $(INCLUDES) $(FLAGS)

linker: main
	otool -L main
