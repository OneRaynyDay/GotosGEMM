#define HAS_CBLAS 1
// #define DYNAMIC

#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xarray.hpp>
#include <xtensor/xrandom.hpp>

#ifdef DYNAMIC
#include "dynamic_gemm.hpp"
#else
#include "static_gemm.hpp"
#endif

#include <iostream>
#include <cstdio>
#include <ctime>
#include <cmath>

#define TIME_IT(EXPR) \
{ \
    std::clock_t __start; \
    double __duration; \
    __start = std::clock(); \
    EXPR; \
    __duration = (std::clock() - __start) / (double) CLOCKS_PER_SEC; \
    std::cout << "Time elapsed : " << __duration << std::endl; \
}

const int X_SIZE = 2001;
const int Y_SIZE = 2103;
const int Z_SIZE = 1907;

template <size_t rows, size_t cols>
void print_matrix(float (&arr)[rows][cols]){
    for(auto i = 0; i < rows; i++){
        for(auto j = 0; j < cols; j++){
            std::cout << arr[i][j] << " ";
        }
        std::cout << std::endl;
    }
}

template <size_t row, size_t mid, size_t col>
void initialize_matrices(float (&a)[row][mid], float (&b)[mid][col], float (&c)[row][col]){
    // Initialize matrices
    for(auto i = 0; i < row; i++){
        for(auto j = 0; j < mid; j++){
            a[i][j] = ((float) (i*Y_SIZE + j)) / (row * mid);
        }
    }
    for(auto i = 0; i < mid; i++){
        for(auto j = 0; j < col; j++){
            b[i][j] = ((float) (i*Z_SIZE + j)) / (mid * col);
        }
    }
    for(auto i = 0; i < row; i++){
        for(auto j = 0; j < col; j++){
            c[i][j] = 0;
        }
    }
}

// naive

template <size_t row, size_t mid, size_t col>
void matmul1(float (&a)[row][mid], float (&b)[mid][col], float (&c)[row][col]){
    for(auto i = 0; i < row; i++){
        for(auto j = 0; j < col; j++){
            float sum = 0;
            for(auto k = 0; k < mid; k++){
                sum += a[i][k] * b[k][j]; 
            }
            c[i][j] = sum;
        }
    } 
}

template <size_t row, size_t col>
bool allclose(float (&a)[row][col], float (&b)[row][col], float threshold = 1e-5, bool verbose = true){
    bool is_equal = true;
    for(auto i = 0; i < row; i++){
        for(auto j = 0; j < col; j++){
            bool current_element = std::abs(a[i][j] - b[i][j]) < threshold;
            if(verbose && !current_element){
                std::cerr << "Element at [" << i << "][" << j << "] is incorrect : " 
                    << a[i][j] << " vs. " << b[i][j] << "." << std::endl;
            }
            is_equal = is_equal && current_element;
        }
    } 
    return is_equal;
}

float a[X_SIZE][Y_SIZE];
float b[Y_SIZE][Z_SIZE];
float c1[X_SIZE][Z_SIZE];
float c2[X_SIZE][Z_SIZE];

int main(){
    std::cout << std::unitbuf;
    initialize_matrices(a, b, c1);
    TIME_IT(matmul1(a, b, c1))
    
    // We must guarrantee c is all zeros at first.
    initialize_matrices(a, b, c2);

#ifdef DYNAMIC
    TIME_IT(gemm::gemm((float* const)a, (float* const) b, (float* const) c2, X_SIZE, Y_SIZE, Z_SIZE))
#else
    TIME_IT(gemm::gemm(a, b, c2))
#endif

    xt::xarray<float> xa = xt::random::randn<float>({X_SIZE, Y_SIZE});
    xt::xarray<float> xb = xt::random::randn<float>({Y_SIZE, Z_SIZE});
    xt::xarray<float> xc;
    TIME_IT(xc = xt::linalg::dot(xa, xb))

    std::cout << allclose(c1, c2, 1e-1, true) << std::endl;
    return 0;
}
