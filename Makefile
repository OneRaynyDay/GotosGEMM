CC=g++-8
INCLUDES=-I${CONDA_PREFIX}/include
# DOCKER_INCLUDES=-I/mnt/home/airbnb/bighead/native/third_party/xtensor/include -I/mnt/home/airbnb/bighead/native/third_party/xsimd/include -I/mnt/home/airbnb/bighead/native/third_party/xtensor-blas/include -I/mnt/home/airbnb/bighead/native/third_party/xtl/include
FLAGS=-std=c++17 -ffast-math -Ofast -mavx2 -lcblas -Wl,-v -g -Wno-vla
DEBUG=-fopt-info-vec-missed 

run: main
	./main

main: main.cpp
	$(CC) -o main main.cpp $(INCLUDES) $(FLAGS)

linker: main
	otool -L main
