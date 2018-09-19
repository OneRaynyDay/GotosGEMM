#include <iostream>
#include <cstdio>
#include <cmath>

namespace gemm{

void print_matrix(float* const arr, size_t rows, size_t cols){
    for(auto i = 0; i < rows; i++){
        for(auto j = 0; j < cols; j++){
            std::cout << arr[i*cols + j] << " ";
        }
        std::cout << std::endl;
    }
}

// = 32 floats
static constexpr size_t BLK_BYTES = 32;
static constexpr size_t BLK_SIZE = BLK_BYTES / 4; 

inline void packb(float* const a, float* const b, size_t m, size_t n, size_t row, size_t col, size_t brow, size_t bcol){
    // size_t m, n in this case means the m,n-th block to pack.
    auto start_row = m * BLK_SIZE;
    auto start_col = n * BLK_SIZE;
    for(size_t i = 0; i < brow; i++){
        for(size_t j = 0; j < bcol; j++){
            b[i*bcol + j] = a[(start_row + i)*col + (start_col + j)];
        }
    }
}

// gemm building blocks

inline void gebp(float* const Ab, float* const Bp, float* const Cp, size_t M, size_t K, size_t N){
    // We can optimize this subroutine but that'd be overkill for now.
    for(auto j = 0; j < N; j++){
        for(auto i = 0; i < M; i++){
            float sum = 0;
            for(auto k = 0; k < K; k++){
                auto _a = Ab[i*K + k];
                auto _b = Bp[k*N + j];
                sum += _a * _b;
            }
            Cp[i*N + j] += sum;
        }
    }
}

inline void gepp(float* const a, float* const Bp, float* const c, size_t p, size_t row, size_t mid, size_t blk_mid, size_t col){
    const size_t M = row / BLK_SIZE;
    const size_t rM = row % BLK_SIZE;

    for(auto i = 0; i < M; i++){
        // Pack A[i*BLK_SIZE : (i+1)*BLK_SIZE][p*BLK_SIZE : (p+1)*BLK_SIZE] into Ab
        float Ab[BLK_SIZE * blk_mid];

        // Reassign C[i*BLK_SIZE : (i+1)*BLK_SIZE][:] into Cp
        float* const Cp = (float* const) &c[i * BLK_SIZE * col];

        packb(a, Ab, i, p, row, mid, BLK_SIZE, blk_mid);

        // The result of Ab and Bp should be in Cp
        gebp(Ab, Bp, Cp, BLK_SIZE, blk_mid, col);
    }
    // Consider residue from the (rM, BLK_SIZE) sub-A multiply with Bp
    if (rM != 0){
        float rAb[rM * blk_mid];
        float* const Cp = (float* const) &c[M * BLK_SIZE * col];
        packb(a, rAb, M, p, row, mid, rM, blk_mid);
        gebp(rAb, Bp, Cp, rM, blk_mid, col);
    }
}

inline void gemm(float* const a, float* const b, float* const c, size_t row, size_t mid, size_t col){
    // Divide up the matrix into panels:
    // Suppose row / BLK_SIZE = M
    //         col / BLK_SIZE = N
    //         mid / BLK_SIZE = K
    //
    // For the rest of the function, we assume A, B, C as a, b, c variables,
    // and {var}p = panel of var
    //     {var}b = block of var
    //
    // If the matrix is not perfectly divisible, then we need to
    // take care of the corresponding edge cases.
    
    // Layout: A = (M,K), B = (K,N), C = (M,N)
    const size_t K = mid / BLK_SIZE;
    const size_t rK = mid % BLK_SIZE;

    for(auto p = 0; p < K; p++){
        // Reassign B[p*BLK_SIZE : (p+1)*BLK_SIZE][:] into Bp
        float* const Bp = (float* const) &b[p * BLK_SIZE * col];
        gepp(a, Bp, c, p, row, mid, BLK_SIZE, col);
    }
    // Consider residue from the (M, rK) sub-A, (rK, N) sub-B
    if (rK != 0){
        float* const Bp = (float* const) &b[K * BLK_SIZE * col];
        gepp(a, Bp, c, K, row, mid, rK, col);
    } 
}
} // namespace gemm
