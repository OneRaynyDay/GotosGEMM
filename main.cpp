#define HAS_CBLAS 1
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xarray.hpp>
#include <xtensor/xrandom.hpp>
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

static constexpr auto X_SIZE = 2003;
static constexpr auto Y_SIZE = 4001;
static constexpr auto Z_SIZE = 4005;
// = 32 floats
static constexpr auto BLK_BYTES = 32;
static constexpr auto BLK_SIZE = BLK_BYTES / 4; 

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

// pack operations

template <size_t row, size_t col>
inline void packp(float (&a)[row][col], float (&b)[BLK_SIZE][col], size_t m){
    // size_t m in this case means the n-th panel to pack, which means
    // m * BLK_SIZE.
    auto start_row = m * BLK_SIZE;
    for(auto i = 0; i < BLK_SIZE; i++){
        for(auto j = 0; j < col; j++){
            b[i][j] = a[start_row + i][j];
        }
    }
}

template <size_t row, size_t col, size_t brow, size_t bcol>
inline void packb(float (&a)[row][col], float (&b)[brow][bcol], size_t m, size_t n){
    // size_t m, n in this case means the m,n-th block to pack.
    auto start_row = m * BLK_SIZE;
    auto start_col = n * BLK_SIZE;
    for(size_t i = 0; i < brow; i++){
        for(size_t j = 0; j < bcol; j++){
            b[i][j] = a[start_row + i][start_col + j];
        }
    }
}

// gemm building blocks

template <size_t M, size_t K, size_t N>
inline void gebp(float (&Ab)[M][K], float (&Bp)[K][N], float (&Cp)[M][N]){
    // We can optimize this subroutine but that'd be overkill for now.  
    for(auto j = 0; j < N; j++){
        for(auto i = 0; i < M; i++){
            float sum = 0;
            for(auto k = 0; k < K; k++){
                sum += Ab[i][k] * Bp[k][j];
            }
            Cp[i][j] += sum;
        }
    }
}

template <size_t row, size_t mid, size_t blk_mid, size_t col>
inline void gepp(float (&a)[row][mid], float (&Bp)[blk_mid][col], float (&c)[row][col], size_t p){
    constexpr size_t M = row / BLK_SIZE;
    constexpr size_t rM = row % BLK_SIZE;

    for(auto i = 0; i < M; i++){
        // Pack A[i*BLK_SIZE : (i+1)*BLK_SIZE][p*BLK_SIZE : (p+1)*BLK_SIZE] into Ab
        float Ab[BLK_SIZE][blk_mid];

        // Reassign C[i*BLK_SIZE : (i+1)*BLK_SIZE][:] into Cp
        float (&Cp)[BLK_SIZE][col] = *(float (*)[BLK_SIZE][col]) &c[i * BLK_SIZE];

        packb(a, Ab, i, p); 
        // The result of Ab and Bp should be in Cp
        gebp(Ab, Bp, Cp);
    } 
    // Consider residue from the (rM, BLK_SIZE) sub-A multiply with Bp
    if constexpr (rM != 0){
        float rAb[rM][blk_mid];
        float (&Cp)[rM][col] = *(float (*)[rM][col]) &c[M * BLK_SIZE];
        packb(a, rAb, M, p); 
        gebp(rAb, Bp, Cp);
    }
}

template <size_t row, size_t mid, size_t col>
inline void gemm(float (&a)[row][mid], float (&b)[mid][col], float (&c)[row][col]){
    // Divide up the matrix into panels:
    // Suppose row / BLK_SIZE = M
    //         col / BLK_SIZE = N
    //         mid / BLK_SIZE = K
    //
    // For the rest of the function, we assume A, B, C as a, b, c variables,
    // and {var}p = panel of var
    //     {var}b = block of var
    //
    // (TODO): If the matrix is not perfectly divisible, then we need to
    // take care of the corresponding edge cases.
    
    // Layout: A = (M,K), B = (K,N), C = (M,N)
    constexpr size_t K = mid / BLK_SIZE;
    constexpr size_t rK = mid % BLK_SIZE;

    for(auto p = 0; p < K; p++){
        // Reassign B[p*BLK_SIZE : (p+1)*BLK_SIZE][:] into Bp
        float (&Bp)[BLK_SIZE][col] = *(float (*)[BLK_SIZE][col]) &b[p * BLK_SIZE];
        gepp(a, Bp, c, p);
    }
    // TODO: Consider residue from the (M, rK) sub-A, (rK, N) sub-B
    if constexpr (rK != 0){
        float (&Bp)[rK][col] = *(float (*)[rK][col]) &b[K * BLK_SIZE];
        gepp(a, Bp, c, K);
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
    initialize_matrices(a, b, c1);
    TIME_IT(matmul1(a, b, c1))
    
    // We must guarrantee c is all zeros at first.
    initialize_matrices(a, b, c2);
    TIME_IT(gemm(a, b, c2))

    xt::xarray<float> xa = xt::random::randn<float>({X_SIZE, Y_SIZE});
    xt::xarray<float> xb = xt::random::randn<float>({Y_SIZE, Z_SIZE});
    xt::xarray<float> xc;
    TIME_IT(xc = xt::linalg::dot(xa, xb))

    std::cout << allclose(c1, c2, 1e-1, true) << std::endl;
    return 0;
}
