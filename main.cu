#include <cuda.h>
#include <stdio.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <mma.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <cooperative_groups/memcpy_async.h>
#include <curand_kernel.h>
#include <iostream>
#include <functional>
#include <array>
#include <set>
#include <unordered_set>
#include <type_traits>
#include <gflags/gflags.h>
#include <nvml.h>
#include <random>
#include <cooperative_groups.h>
#include <cuda/barrier>
#include <cuda/atomic>
#include <cuda/pipeline>
#include <x86intrin.h>
#include <cuda_bf16.h>
#include <cassert>

#include "macro.cuh"
#include "bitnet_kernels.cuh"
#include "AsyncCopy_PTX.cuh"

namespace cg = cooperative_groups;
using barrier = cuda::barrier<cuda::thread_scope_block>;
using pipeline = cuda::pipeline<cuda::thread_scope_block>;
using atomicBool = cuda::atomic<bool, cuda::thread_scope_block>;
typedef unsigned char uchar;

DEFINE_bool(cu_blas, false, "Run cuBLAS method when true");
DEFINE_uint64(row_split3_small, 0, "Run Row-wise SplitS 3 small method with specified split size");
DEFINE_uint64(row_split_delta2, 0, "Run Row-wise Split Delta method 2 with specified split size");
DEFINE_bool(i2s, false, "Run I2_S method when true");

DEFINE_uint64(M, 32L, "M"); // batch size
DEFINE_uint64(K, 3072L, "K"); // hidden size
DEFINE_uint64(N, 768L, "N"); // hidden size * 4
DEFINE_uint64(S, 256, "Number of non-zero values in a column of W. Sparsity = K / S");
DEFINE_uint64(L, 1L, "Number of how each CUDA thread calculates in row-wise method");

DEFINE_uint32(iter, 10, "Number of launching kernels");

// X: MxK  W: KxN  C: MxN
#define M FLAGS_M
#define K FLAGS_K
#define N FLAGS_N
#define S FLAGS_S
#define L FLAGS_L

// Please define from compiler option
//#define X_MAJOR MAJOR_ROW
//#define W_MAJOR MAJOR_ROW
//#define C_MAJOR MAJOR_ROW
#define MAJOR_STR(m) (m == MAJOR_ROW ? "ROW" : "COL")

struct ctx{
    uint64_t m;
    uint64_t k;
    uint64_t n;
    uint64_t s;
    uint64_t l;
    u_int64_t b;
} ctx_v;


template <std::size_t N>
struct Unroller {
    template <typename T>
    __device__ static void Execute(T&& t) {
        if constexpr (N > 0) {
            Unroller<N-1>::Execute(t);
            t(std::integral_constant<std::size_t, N-1>{});
        }
    }
};

void init_ctx(){
    //assert(M % 16 == 0 && "mod 16 should be 0"); // not need to be that when using i2s mathod
    assert(K % 16 == 0 && "mod 16 should be 0");
    assert(N % 16 == 0 && "mod 16 should be 0");
    assert(K < 65536 && "K should be fit in the maximum of short, since W_M manages indices with short type");
    assert(S % 2 == 0 && "S mod 2 must be 0 s.t. W_map and W_negative becomes the same size");
    assert(K > S && "K should be bigger than S");

    ctx_v = {
            .m = M,
            .k = K,
            .n = N,
            .s = S,
            .l = L,
            .b = 8 // for performance
    };
}

static const char *_cudaGetErrorEnum(cudaError_t error) {
    return cudaGetErrorName(error);
}

template <typename T>
void check(T result, char const *const func, const char *const file,
           int const line) {
    if (result) {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
                static_cast<unsigned int>(result), _cudaGetErrorEnum(result), func);
        exit(EXIT_FAILURE);
    }
}
#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

#define checkKernelErrors(expr)                             \
  do {                                                      \
    expr;                                                   \
                                                            \
    cudaError_t __err = cudaGetLastError();                 \
    if (__err != cudaSuccess) {                             \
      printf("Line %d: '%s' failed: %s\n", __LINE__, #expr, \
             cudaGetErrorString(__err));                    \
      abort();                                              \
    }                                                       \
  } while (0)

void checkCublasError(cublasStatus_t status, int line)
{
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("Cublas Error at line %d, Error Code: %d\n", line, status);
        exit(EXIT_FAILURE);
    }
}

__global__ void char2f16(const char* in, float* out){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    out[i] = static_cast<float>(in[i]);
}

__global__ void f162int(const float* in, int* out){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    out[i] = static_cast<int>(in[i]);
}

__global__ void prepareW_i2s(int8_t *W_i2s, ctx ctx){
u_int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
u_int64_t col = tid;
if(col >= ctx.n){
// this thread won't work for init
return;
}

char tmp[16];
for(int i = 0; i < 16; i++){
tmp[i] = 0;
}
for(int i = 0; i < 4; i++){
tmp[i] = -1;
}
for(int i = 0; i < 4; i++){
tmp[i] = 1;
}

uint32_t seed = (uint32_t) (0xCAFEBABE ^ (col / 32));

for(int i = 16 - 1; i > 0; --i){
int j = xorshift32(&seed) % (i + 1);
char t = tmp[i];
tmp[i] = tmp[j];
tmp[j] = t;
}

// +2
for(int i = 0; i < 16; i++){
tmp[i] += 2;
}
char shuffle_tmp[16];
for(int i = 0; i < 4; i++){
for(int j = 0; j < 4; j++){
int in_pos = 3 - i;
int out_pos = 3 - j;
shuffle_tmp[out_pos * 4 + in_pos] = tmp[i * 4 + j];
}
}
// to int32
int code = 0;
for(int i = 0; i < 16; i++){
code |= shuffle_tmp[i] << (30 - i*2);
}

for(unsigned i = 0; i < ctx.k; i+=16){
reinterpret_cast<int *>(&AT(MAJOR_COL) (W_i2s, ctx.k / 4, ctx.n, i/4, col))[0] = code;
}
}

/**
 * Prepare both W_mat and W_map before the measurement.
 */
__global__ void prepareW_map(
        char * W, // working memory
        unsigned short* const W_map,
        unsigned short* const W_map_negative,
        unsigned short* const W_map_split_base,
        unsigned short* const W_map_split_negative_base,
        unsigned char* const W_map_split_delta,
        unsigned char* const W_map_split_negative_delta,
        unsigned char* const W_map_delta2_d,
        unsigned char* const W_map_negative_delta2_d,
        unsigned char* const W_map_delta_warp_acc,
        unsigned char* const W_map_negative_delta_warp_acc,
        char* const W_vcsc_values,
        unsigned short* const W_vcsc_counts,
        unsigned short* const W_vcsc_indices,
        unsigned short* const W_vcsc_indices_div16,

        unsigned short* const W_map_32_div,
        unsigned short* const W_map_negative_32_div,
        unsigned short* const W_map_8_div,
        unsigned short* const W_map_negative_8_div,
        unsigned short* const W_map_16_div,
        unsigned short* const W_map_negative_16_div,
        unsigned short* const W_map_64_div,
        unsigned short* const W_map_negative_64_div,
        unsigned short* const W_map_128_div,
        unsigned short* const W_map_negative_128_div,
        unsigned short* const W_map_256_div,
        unsigned short* const W_map_negative_256_div,
        unsigned short* const W_map_512_div,
        unsigned short* const W_map_negative_512_div,

        unsigned char* const W_map_delta2_div256,
        unsigned char* const W_map_negative_delta2_div256,
        unsigned char* const W_map_delta2_div32,
        unsigned char* const W_map_negative_delta2_div32,
        ctx ctx){
    u_int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    if(tid >= ctx.n){
        // this thread won't work for init
        return;
    }

    u_int64_t col = tid;

    for(int i = 0; i < ctx.k; i++){
        AT(W_MAJOR) (W, ctx.k, ctx.n, i, col) = 0;
    }
    for(int i = 0; i < ctx.s / 2; i++){
        AT(W_MAJOR) (W, ctx.k, ctx.n, i, col) = -1;
    }
    for(int i = ctx.s / 2; i < ctx.s; i++){
        AT(W_MAJOR) (W, ctx.k, ctx.n, i, col) = 1;
    }

    uint32_t seed = (uint32_t) (0xCAFEBABE ^ col);

    for(int i = ctx.k - 1; i > 0; --i){
        int j = xorshift32(&seed) % (i + 1);
        char tmp = AT(W_MAJOR) (W, ctx.k, ctx.n, i, col);
        AT(W_MAJOR) (W, ctx.k, ctx.n, i, col) = AT(W_MAJOR) (W, ctx.k, ctx.n, j, col);
        AT(W_MAJOR) (W, ctx.k, ctx.n, j, col) = tmp;
    }

    int count_1 = 0, count_m1 = 0;
    for(int i = 0; i < ctx.k; i++){
        if(AT(W_MAJOR) (W, ctx.k, ctx.n, i, col) == 1){
            AT(W_MAJOR) (W_map, ctx.s / 2, ctx.n, count_1++, col) = i;
        }else if (AT(W_MAJOR) (W, ctx.k, ctx.n, i, col) == -1){
            AT(W_MAJOR) (W_map_negative, ctx.s / 2, ctx.n, count_m1++, col) = i;
        }
    }

    assert(count_1 == ctx.s/ 2 && count_m1 == ctx.s / 2 && "W matrix corrupt");

    __syncthreads();

    if(ctx.s % 1024 != 0){
#ifndef NDEBUG
        if(tid == 0) {
            printf("WARNING! do not use W_map_512_div\n");
        }
#endif
    }else{
        // See matrix as [512*N, S/1024]
        for(int i = tid * 512; i < tid * 512 + 512; i++){
            for(int j = 0; j < ctx.s / 1024; j++){
                int original_row = j * 512 + i % 512;
                int original_col = i / 512;
                        AT(MAJOR_COL)(W_map_512_div, 512 * ctx.n, ctx.s / 1024, i, j) = AT(W_MAJOR) (W_map, ctx.s / 2, ctx.n, original_row, original_col);
                        AT(MAJOR_COL)(W_map_negative_512_div, 512 * ctx.n, ctx.s / 1024, i, j) = AT(W_MAJOR) (W_map_negative, ctx.s / 2, ctx.n, original_row, original_col);
            }
        }
    }

    //assert(ctx.s % 512 == 0 && "Wrong S setting");
    // See matrix as [256*N, S/512]
    int div256_cols = CEIL_DIV( (ctx.s /2), 256 );

    for(int i = tid * 256; i < tid * 256 + 256; i++){
        for(int j = 0; j < div256_cols; j++){
            int original_row = j * 256 + i % 256;
            int original_col = i / 256;

            if(original_row < ctx.s / 2){
                        AT(MAJOR_COL)(W_map_256_div, 256 * ctx.n, div256_cols, i, j) = AT(W_MAJOR) (W_map, ctx.s / 2, ctx.n, original_row, original_col);
                        AT(MAJOR_COL)(W_map_negative_256_div, 256 * ctx.n, div256_cols, i, j) = AT(W_MAJOR) (W_map_negative, ctx.s / 2, ctx.n, original_row, original_col);
            }
        }
    }

    if(ctx.s % 256 != 0){
#ifndef NDEBUG
        if(tid == 0) {
            printf("WARNING! do not use W_map_128_div\n");
        }
#endif
    }else{
        // See matrix as [128*N, S/256]
        for(int i = tid * 128; i < tid * 128 + 128; i++){
            for(int j = 0; j < ctx.s / 256; j++){
                int original_row = j * 128 + i % 128;
                int original_col = i / 128;
                        AT(MAJOR_COL)(W_map_128_div, 128 * ctx.n, ctx.s / 256, i, j) = AT(W_MAJOR) (W_map, ctx.s / 2, ctx.n, original_row, original_col);
                        AT(MAJOR_COL)(W_map_negative_128_div, 128 * ctx.n, ctx.s / 256, i, j) = AT(W_MAJOR) (W_map_negative, ctx.s / 2, ctx.n, original_row, original_col);
            }
        }
    }

    assert(ctx.s % 128 == 0 && "Wrong S setting");
    // See matrix as [64*N, S/128]
    for(int i = tid * 64; i < tid * 64 + 64; i++){
        for(int j = 0; j < ctx.s / 128; j++){
            int original_row = j * 64 + i % 64;
            int original_col = i / 64;
                    AT(MAJOR_COL)(W_map_64_div, 64 * ctx.n, ctx.s / 128, i, j) = AT(W_MAJOR) (W_map, ctx.s / 2, ctx.n, original_row, original_col);
                    AT(MAJOR_COL)(W_map_negative_64_div, 64 * ctx.n, ctx.s / 128, i, j) = AT(W_MAJOR) (W_map_negative, ctx.s / 2, ctx.n, original_row, original_col);
        }
    }

    assert(ctx.s % 64 == 0 && "Wrong S setting");
    // See matrix as [32*N, S/64]
    for(int i = tid * 32; i < tid * 32 + 32; i++){
        for(int j = 0; j < ctx.s / 64; j++){
            int original_row = j * 32 + i % 32;
            int original_col = i / 32;
                    AT(MAJOR_COL)(W_map_32_div, 32 * ctx.n, ctx.s / 64, i, j) = AT(W_MAJOR) (W_map, ctx.s / 2, ctx.n, original_row, original_col);
                    AT(MAJOR_COL)(W_map_negative_32_div, 32 * ctx.n, ctx.s / 64, i, j) = AT(W_MAJOR) (W_map_negative, ctx.s / 2, ctx.n, original_row, original_col);
        }
    }

    assert(ctx.s % 16 == 0 && "Wrong S setting");
    // See matrix as [16*N, S/32]
    for(int i = tid * 16; i < tid * 16 + 16; i++){
        for(int j = 0; j < ctx.s / 32; j++){
            int original_row = j * 16 + i % 16;
            int original_col = i / 16;
                    AT(MAJOR_COL)(W_map_16_div, 16 * ctx.n, ctx.s / 32, i, j) = AT(W_MAJOR) (W_map, ctx.s / 2, ctx.n, original_row, original_col);
                    AT(MAJOR_COL)(W_map_negative_16_div, 16 * ctx.n, ctx.s / 32, i, j) = AT(W_MAJOR) (W_map_negative, ctx.s / 2, ctx.n, original_row, original_col);
        }
    }

    assert(ctx.s % 16 == 0 && "Wrong S setting");
    // See matrix as [8*N, S/16]
    for(int i = tid * 8; i < tid * 8 + 8; i++){
        for(int j = 0; j < ctx.s / 16; j++){
            int original_row = j * 8 + i % 8;
            int original_col = i / 8;
                    AT(MAJOR_COL)(W_map_8_div, 8 * ctx.n, ctx.s / 16, i, j) = AT(W_MAJOR) (W_map, ctx.s / 2, ctx.n, original_row, original_col);
                    AT(MAJOR_COL)(W_map_negative_8_div, 8 * ctx.n, ctx.s / 16, i, j) = AT(W_MAJOR) (W_map_negative, ctx.s / 2, ctx.n, original_row, original_col);
        }
    }

    for(int i = 0; i < ctx.s/2; i++){
        if(i==0){
            AT(W_MAJOR)(W_map_delta2_d, ctx.s / 2, ctx.n, i, col)
                    = AT(W_MAJOR) (W_map, ctx.s / 2, ctx.n, i, col);
        }else{
            AT(W_MAJOR)(W_map_delta2_d, ctx.s / 2, ctx.n, i, col)
                    = AT(W_MAJOR) (W_map, ctx.s / 2, ctx.n, i, col) - AT(W_MAJOR) (W_map, ctx.s / 2, ctx.n, i-1, col);
        }
    }

    for(int i = 0; i < ctx.s/2; i++){
        if(i==0){
            AT(W_MAJOR)(W_map_negative_delta2_d, ctx.s / 2, ctx.n, i, col)
                    = AT(W_MAJOR) (W_map_negative, ctx.s / 2, ctx.n, i, col);
        }else{
            AT(W_MAJOR)(W_map_negative_delta2_d, ctx.s / 2, ctx.n, i, col)
                    = AT(W_MAJOR) (W_map_negative, ctx.s / 2, ctx.n, i, col) - AT(W_MAJOR) (W_map_negative, ctx.s / 2, ctx.n, i-1, col);
        }
    }

    __syncthreads();



    div256_cols = CEIL_DIV( (ctx.s /2), 256 );

    // See matrix as [256*N, S/512]
    for(int i = tid * 256; i < tid * 256 + 256; i++){
        for(int j = 0; j < div256_cols; j++){
            int original_row = j * 256 + i % 256;
            int original_col = i / 256;

            if(original_row < ctx.s / 2){
                        AT(MAJOR_COL)(W_map_delta2_div256, 256 * ctx.n, div256_cols, i, j) = AT(W_MAJOR) (W_map_delta2_d, ctx.s / 2, ctx.n, original_row, original_col);
                        AT(MAJOR_COL)(W_map_negative_delta2_div256, 256 * ctx.n, div256_cols, i, j) = AT(W_MAJOR) (W_map_negative_delta2_d, ctx.s / 2, ctx.n, original_row, original_col);
            }
        }
    }


    assert(ctx.s % 64 == 0 && "Wrong S setting");
    // See matrix as [32*N, S/64]
    for(int i = tid * 32; i < tid * 32 + 32; i++){
        for(int j = 0; j < ctx.s / 64; j++){
            int original_row = j * 32 + i % 32;
            int original_col = i / 32;
                    AT(MAJOR_COL)(W_map_delta2_div32, 32 * ctx.n, ctx.s / 64, i, j) = AT(W_MAJOR) (W_map_delta2_d, ctx.s / 2, ctx.n, original_row, original_col);
                    AT(MAJOR_COL)(W_map_negative_delta2_div32, 32 * ctx.n, ctx.s / 64, i, j) = AT(W_MAJOR) (W_map_negative_delta2_d, ctx.s / 2, ctx.n, original_row, original_col);
        }
    }

    for(int i = 0; i < ctx.s/2; i+=32){
        AT(W_MAJOR)(W_map_delta_warp_acc, ctx.s / 2, ctx.n, i, col)
                = AT(W_MAJOR) (W_map_delta2_d, ctx.s / 2, ctx.n, i, col); // assign first delta
        ushort acc = AT(W_MAJOR) (W_map_delta2_d, ctx.s / 2, ctx.n, i, col);
        for(int j = 1; j < 32; j++){
            acc += AT(W_MAJOR) (W_map_delta2_d, ctx.s / 2, ctx.n, i+j, col);
#ifndef NDEBUG
            if(acc >= 255){
                if(tid == 0){
                    printf("WARNING! acc >= 255! W_map_delta_warp_acc fails!\n");
                }
            }
#endif
            AT(W_MAJOR)(W_map_delta_warp_acc, ctx.s / 2, ctx.n, i+j, col) = acc;
        }
    }

    for(int i = 0; i < ctx.s/2; i+=32){
        AT(W_MAJOR)(W_map_negative_delta_warp_acc, ctx.s / 2, ctx.n, i, col)
                = AT(W_MAJOR) (W_map_negative_delta2_d, ctx.s / 2, ctx.n, i, col); // assign first delta
        ushort acc = AT(W_MAJOR) (W_map_negative_delta2_d, ctx.s / 2, ctx.n, i, col);
        for(int j = 1; j < 32; j++){
            acc += AT(W_MAJOR) (W_map_negative_delta2_d, ctx.s / 2, ctx.n, i+j, col);
#ifndef NDEBUG
            if(acc >= 255){
                if(tid == 0){
                    printf("WARNING! acc >= 255! W_map_negative_delta_warp_acc fails!\n");
                }
            }
#endif
            AT(W_MAJOR)(W_map_negative_delta_warp_acc, ctx.s / 2, ctx.n, i+j, col) = acc;
        }
    }

    AT(W_MAJOR)(W_vcsc_values, 2, ctx.n, 0, col) = 1;
    AT(W_MAJOR)(W_vcsc_values, 2, ctx.n, 1, col) = -1;

    AT(W_MAJOR)(W_vcsc_counts, 2, ctx.n, 0, col) = ctx.s / 2;
    AT(W_MAJOR)(W_vcsc_counts, 2, ctx.n, 1, col) = ctx.s / 2;

    int vcsc_count_1 = 0, vcsc_count_m1 = ctx.s / 2;
    for(int i = 0; i < ctx.k; i++){
        if(AT(W_MAJOR) (W, ctx.k, ctx.n, i, col) == 1){
            AT(W_MAJOR) (W_vcsc_indices, ctx.s, ctx.n, vcsc_count_1, col) = i;
            vcsc_count_1++;
        }else if (AT(W_MAJOR) (W, ctx.k, ctx.n, i, col) == -1){
            AT(W_MAJOR) (W_vcsc_indices, ctx.s, ctx.n, vcsc_count_m1, col) = i;
            vcsc_count_m1++;
        }
    }

    __syncthreads();

    // See matrix as [16*N, S/16]
    for(int i = tid * 16; i < tid * 16 + 16; i++){
        for(int j = 0; j < ctx.s / 16; j++){
            int original_row = j * 16 + i % 16;
            int original_col = i / 16;
                    AT(MAJOR_COL)(W_vcsc_indices_div16, 16 * ctx.n, ctx.s / 16, i, j) =
                    AT(W_MAJOR) (W_vcsc_indices, ctx.s, ctx.n, original_row, original_col);
        }
    }
}

__device__ inline void int_to_shorts_le(int32_t val, uint16_t shorts[2]) {
    // lower 16 bits in shorts[0], upper 16 bits in shorts[1]
    shorts[0] = (uint16_t)( val        & 0xFFFF );
    shorts[1] = (uint16_t)((val >> 16) & 0xFFFF );
}

__device__ inline void int_to_bytes_le(int32_t val, char bytes[4]) {
    bytes[0] = (char)((val >>  0) & 0xFF); // 最下位バイト
    bytes[1] = (char)((val >>  8) & 0xFF);
    bytes[2] = (char)((val >> 16) & 0xFF);
    bytes[3] = (char)((val >> 24) & 0xFF); // 最上位バイト
}

__device__ ushort2 int_to_ushort2(int val) {
    unsigned short val_x = (unsigned short)(val & 0xFFFF);      // 下位16ビット
    unsigned short val_y = (unsigned short)((val >> 16) & 0xFFFF); // 上位16ビット
    return make_ushort2(val_x, val_y);
}


template <const uint M_, const uint K_, const uint N_, const uint S_, const uint SPLIT> // 32~256
__global__ void __launch_bounds__(SPLIT) rowWiseSplit3Coalesced(
        const char* __restrict__ const X,
        const unsigned short* __restrict__ const W_map,
        const unsigned short* __restrict__ const W_map_negative,
        int* __restrict__ const C){

    static_assert((S_ / 2) % SPLIT == 0, "Should be completely devidable");

    const uint col = blockIdx.x;
    const uint m_row = blockIdx.y;

    // This makes less IMAD operation
    const ushort *W_map_base = &W_map[SPLIT * blockIdx.x];
    const ushort *W_map_neg_base = &W_map_negative[SPLIT * blockIdx.x];
    const char *X_base = &X[m_row * K_]; // must be row-major

    int accum = 0;

    // NOTE: we need to adjust unroll ratio explicitly to get the best performance
    static constexpr int UNROLL_FACTOR = ((S_ / 2) / SPLIT);
    //#pragma unroll 16
#pragma unroll UNROLL_FACTOR
    for(uint i = threadIdx.x; i < (S_ / 2) * N_ ; i+=SPLIT * N_){
        accum += __ldg(&X_base[W_map_base[i]]);
        accum -= __ldg(&X_base[W_map_neg_base[i]]);
    }

    __shared__ int sbuf[SPLIT / 32];
    accum = __reduce_add_sync(0xffffffff, accum);
    if((threadIdx.x & 31) == 0){
        sbuf[threadIdx.x >> 5] = accum;
        __syncthreads();

        if(threadIdx.x == 0){
            accum = 0;
#pragma unroll
            for(char i = 0; i < SPLIT / 32; i++){
                accum += sbuf[i];
            }
            C[m_row * N_ + col] = accum;
        }
    }
}

template <const uint M_, const uint K_, const uint N_, const uint S_, const uint SPLIT = 32>
__global__ void __launch_bounds__(32) rowWiseSplit3Small4(
        const char* __restrict__ const X,
        const unsigned short* __restrict__ const W_map,
        const unsigned short* __restrict__ const W_map_negative,
        int* __restrict__ const C){

    static constexpr int COLS_PER_WARP = 32 / SPLIT; // 1, 4
    static_assert((S_ / 2) % (SPLIT) == 0, "Wrong SPLIT Size");

    const uint c_col = blockIdx.x * COLS_PER_WARP + threadIdx.y;
    const uint m_row = blockIdx.y;
    // assume size of thread block =< 32
    const int lane_id = threadIdx.x
                        + threadIdx.y * blockDim.x
                        + threadIdx.z * blockDim.x * blockDim.y;

    const ushort *W_map_base = &W_map[32 * blockIdx.x];
    const ushort *W_map_neg_base = &W_map_negative[32 * blockIdx.x];
    const char *X_base = &X[m_row * K_]; // must be row-major

    int accum = 0;

    static constexpr int UNROLL_FACTOR = ((S_ / 2) / SPLIT);
#pragma unroll UNROLL_FACTOR
    for(uint i = lane_id; i < ((S_ / 2) / SPLIT) * SPLIT * N_ ; i+= SPLIT * N_){
        accum += __ldg(&X_base[W_map_base[i]]);
        accum -= __ldg(&X_base[W_map_neg_base[i]]);
    }

#pragma unroll
    for(int i = SPLIT / 2; i > 0; i /= 2){
        accum += __shfl_down_sync(0xffffffff, accum, i, SPLIT);
    }

    if (threadIdx.x == 0) {
        __stwt(&C[m_row * N_ + c_col], accum);
    }
}


// only for blocksize <= 32
// Use Unroll
template <const uint M_, const uint K_, const uint N_, const uint S_, const uint SPLIT = 32, typename COPY_TYPE = int2>
__global__ void __launch_bounds__(SPLIT) rowWiseSplit2DeltaSmallAsync2(
        const char* __restrict__ const X,
        const unsigned char* __restrict__ const W_map_delta,
        const unsigned char* __restrict__ const W_map_negative_delta,
        int* __restrict__ const C){


    static_assert((S_ / 2) % SPLIT == 0, "Should be completely devidable");
    static constexpr int COLS_PER_WARP = 32 / SPLIT; // 1
    static constexpr int W_COPY_BYTE_SIZE = sizeof(COPY_TYPE); // 8
    static constexpr int W_COPY_NUMS = 32 * W_COPY_BYTE_SIZE / sizeof(unsigned char); // 256
    static constexpr int W_COPY_SKIP_NUM = W_COPY_BYTE_SIZE / sizeof(unsigned char); // 8

    const uint c_col = blockIdx.x * COLS_PER_WARP + threadIdx.y;
    const uint m_row = blockIdx.y;

    // This makes less IMAD operation
    const unsigned char *W_map_delta_base = &W_map_delta[W_COPY_NUMS * blockIdx.x];
    const unsigned char *W_map_neg_delta_base = &W_map_negative_delta[W_COPY_NUMS * blockIdx.x];
    const char *X_base = &X[m_row * K_]; // must be row-major

    int accum = 0;

    static_assert((S_ / 2) % W_COPY_NUMS == 0, "Should be completely devidable");
    static constexpr int BATCH_SIZE = (S_ / 2) / W_COPY_NUMS;
    __shared__ __align__(W_COPY_BYTE_SIZE) uchar w_pos_delta_buf[S_/2];
    __shared__ __align__(W_COPY_BYTE_SIZE) uchar w_neg_delta_buf[S_/2];

    const int lane_id = threadIdx.x % 32;
    uint j = 0;
    ushort next_iter_pos = 0;
    ushort next_iter_neg = 0;

#pragma unroll BATCH_SIZE
    for(uint i = lane_id * W_COPY_SKIP_NUM; i < (S_ / 2); i += W_COPY_NUMS){ // ushort coordinate in w_buf
        cp_async<W_COPY_BYTE_SIZE>(reinterpret_cast<COPY_TYPE *>(&w_pos_delta_buf[i]),
                                   reinterpret_cast<const COPY_TYPE *>(&W_map_delta_base[j + lane_id * W_COPY_SKIP_NUM]));
        cp_async<W_COPY_BYTE_SIZE>(reinterpret_cast<COPY_TYPE *>(&w_neg_delta_buf[i]),
                                   reinterpret_cast<const COPY_TYPE *>(&W_map_neg_delta_base[j + lane_id * W_COPY_SKIP_NUM]));

        cp_async_group_commit();
        j += W_COPY_NUMS * N_;
    }

    Unroller<BATCH_SIZE>::Execute([&](auto I) {
        constexpr std::size_t i = I();
        cp_async_wait_group<i>();

#pragma unroll
        for(uint j = threadIdx.x % 32; j < W_COPY_NUMS; j+=32){
            ushort delta_pos = w_pos_delta_buf[i*W_COPY_NUMS+j];
            ushort delta_neg = w_neg_delta_buf[i*W_COPY_NUMS+j];

#pragma unroll
            for(uint offset = 1; offset < 32; offset*=2){
                uint32_t packed = __shfl_up_sync(0xffffffff, ((uint32_t)delta_neg << 16) | (uint32_t)delta_pos, offset);

                delta_pos += (lane_id >= offset) ? (ushort)(packed & 0xffff) : 0;
                delta_neg += (lane_id >= offset) ? (ushort)((packed >> 16) & 0xffff) : 0;
            }

            delta_pos += next_iter_pos;
            delta_neg += next_iter_neg;

            next_iter_pos = __shfl_sync(0xffffffff, delta_pos, 31);
            next_iter_neg = __shfl_sync(0xffffffff, delta_neg, 31);

            accum += __ldg(&X_base[delta_pos]);
            accum -= __ldg(&X_base[delta_neg]);
        }
    });

    accum = __reduce_add_sync(0xffffffff, accum);

    if (threadIdx.x == 0) {
        __stwt(&C[m_row * N_ + c_col], accum);
    }
}



// only for blocksize <= 32
// Use Unroll
// suoppot when (S_ / 2) % W_COPY_NUMS != 0
template <const uint M_, const uint K_, const uint N_, const uint S_, const uint SPLIT = 32, typename COPY_TYPE = int2>
__global__ void __launch_bounds__(SPLIT) rowWiseSplit2DeltaSmallAsync3(
        const char* __restrict__ const X,
        const unsigned char* __restrict__ const W_map_delta,
        const unsigned char* __restrict__ const W_map_negative_delta,
        int* __restrict__ const C){


    static_assert((S_ / 2) % SPLIT == 0, "Should be completely devidable");
    static constexpr int COLS_PER_WARP = 32 / SPLIT; // 1
    static constexpr int W_COPY_BYTE_SIZE = sizeof(COPY_TYPE); // 8
    static constexpr int W_COPY_NUMS = 32 * W_COPY_BYTE_SIZE / sizeof(unsigned char); // 256
    static constexpr int W_COPY_SKIP_NUM = W_COPY_BYTE_SIZE / sizeof(unsigned char); // 8

    const uint c_col = blockIdx.x * COLS_PER_WARP + threadIdx.y;
    const uint m_row = blockIdx.y;

    // This makes less IMAD operation
    const unsigned char *W_map_delta_base = &W_map_delta[W_COPY_NUMS * blockIdx.x];
    const unsigned char *W_map_neg_delta_base = &W_map_negative_delta[W_COPY_NUMS * blockIdx.x];
    const char *X_base = &X[m_row * K_]; // must be row-major

    int accum = 0;

    static constexpr int BATCH_SIZE = CEIL_DIV( (S_ / 2), W_COPY_NUMS );
    __shared__ __align__(W_COPY_BYTE_SIZE) uchar w_pos_delta_buf[S_/2];
    __shared__ __align__(W_COPY_BYTE_SIZE) uchar w_neg_delta_buf[S_/2];

    const int lane_id = threadIdx.x % 32;
    uint j = 0;
    ushort next_iter_pos = 0;
    ushort next_iter_neg = 0;

#pragma unroll BATCH_SIZE
    for(uint i = lane_id * W_COPY_SKIP_NUM; i < (S_ / 2); i += W_COPY_NUMS){ // ushort coordinate in w_buf
        cp_async<W_COPY_BYTE_SIZE>(reinterpret_cast<COPY_TYPE *>(&w_pos_delta_buf[i]),
                                   reinterpret_cast<const COPY_TYPE *>(&W_map_delta_base[j + lane_id * W_COPY_SKIP_NUM]));
        cp_async<W_COPY_BYTE_SIZE>(reinterpret_cast<COPY_TYPE *>(&w_neg_delta_buf[i]),
                                   reinterpret_cast<const COPY_TYPE *>(&W_map_neg_delta_base[j + lane_id * W_COPY_SKIP_NUM]));

        cp_async_group_commit();
        j += W_COPY_NUMS * N_;
    }

    Unroller<BATCH_SIZE>::Execute([&](auto I) {
        constexpr std::size_t i = I();
        cp_async_wait_group<i>();

#pragma unroll
        for(uint j = threadIdx.x % 32; j < W_COPY_NUMS; j+=32){

            if((i*W_COPY_NUMS+j) < S_ / 2 ){
                ushort delta_pos = w_pos_delta_buf[i*W_COPY_NUMS+j];
                ushort delta_neg = w_neg_delta_buf[i*W_COPY_NUMS+j];

#pragma unroll
                for(uint offset = 1; offset < 32; offset*=2){
                    uint32_t packed = __shfl_up_sync(0xffffffff, ((uint32_t)delta_neg << 16) | (uint32_t)delta_pos, offset);

                    delta_pos += (lane_id >= offset) ? (ushort)(packed & 0xffff) : 0;
                    delta_neg += (lane_id >= offset) ? (ushort)((packed >> 16) & 0xffff) : 0;
                }

                delta_pos += next_iter_pos;
                delta_neg += next_iter_neg;

                next_iter_pos = __shfl_sync(0xffffffff, delta_pos, 31);
                next_iter_neg = __shfl_sync(0xffffffff, delta_neg, 31);

                accum += __ldg(&X_base[delta_pos]);
                accum -= __ldg(&X_base[delta_neg]);
            }
        }
    });

    accum = __reduce_add_sync(0xffffffff, accum);

    if (threadIdx.x == 0) {
        __stwt(&C[m_row * N_ + c_col], accum);
    }
}


float measureKernel(std::function<void(void)> fn){
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    fn();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return milliseconds;
}

/**
 * Using uniform distribution to make the range from -127 to 127!
 */
void make_X(char *X){
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dist(0.0, 5.0);

    for(unsigned i = 0; i < M * K; i++) {
        int value = static_cast<int>(std::round(dist(gen)));
        value = std::clamp(value, -127, 127);
        X[i] = value;
    }
}

int main(int argc, char** argv){
    gflags::SetUsageMessage("matrix multiply speed check");
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    init_ctx();

    char *X_d;
    cudaMalloc((void**) &X_d, sizeof(char) * M * K);
    char *X_d_ext;  // extend
    cudaMalloc((void**) &X_d_ext, CEIL_DIV(M * K, 512) * 512);

    char *X_ar;
    posix_memalign((void**) &X_ar, 16, sizeof(char) * M * K); make_X(X_ar);
    cudaMemcpy(X_d, X_ar, sizeof(char) * M * K, cudaMemcpyHostToDevice);

    std::cout << "make_X done" << std::endl;

    int *c_d; cudaMalloc((void**)  &c_d, sizeof(int) * M * N ); cudaMemset(c_d, 0, sizeof(int) * M * N);

    printf("Compiled on %s at %s\n", __DATE__, __TIME__);

    std::cout
            << "Start: "
            << "M=" << M
            << " K=" << K
            << " N=" << N
            << " ITER=" << FLAGS_iter
            << " S=" << ctx_v.s
            << " (" << (100.0 - 100.0 * ((float) S / (float) K)) << "% Sparsity)"
            << " L=" << L
            << " B=" << ctx_v.b
            << " X_MAJOR=" << MAJOR_STR(X_MAJOR)
            << " W_MAJOR=" << MAJOR_STR(W_MAJOR)
            << " C_MAJOR=" << MAJOR_STR(C_MAJOR)
            << std::endl;


    float ms = 0;

    char *W_d = nullptr;




    if(FLAGS_cu_blas) {
        STRICT_ASSERT(X_MAJOR == MAJOR_COL, "Wrong layout");
        STRICT_ASSERT(W_MAJOR == MAJOR_COL, "Wrong layout");
        STRICT_ASSERT(C_MAJOR == MAJOR_COL, "Wrong layout");

        cublasHandle_t handle;
        cublasCreate(&handle);
        cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);
        cudaDeviceSynchronize();

        int alpha = 1, beta = 0;

        ms = measureKernel([&]() {
            for (size_t i = 0; i < FLAGS_iter; i++) {
                cublasStatus_t cublas_status = cublasGemmEx(
                        handle,
                        CUBLAS_OP_N,
                        CUBLAS_OP_N,
                        M, N, K,
                        &alpha,
                        X_d, CUDA_R_8I, M,
                        W_d, CUDA_R_8I, K,
                        &beta,
                        c_d, CUDA_R_32I, M,
                        CUBLAS_COMPUTE_32I,
                        CUBLAS_GEMM_DEFAULT);
                checkCublasError(cublas_status, __LINE__);
                checkCudaErrors(cudaDeviceSynchronize());
            }
        });
        std::cout << ms / ((float) FLAGS_iter) << std::endl;
    }

    char *W_work_d;
    unsigned short *W_map_d;
    unsigned short *W_map_negative_d;
    unsigned short *W_map_split_base_d;
    unsigned short *W_map_split_negative_base_d;
    unsigned char  *W_map_split_delta_d;
    unsigned char  *W_map_split_negative_delta_d;
    unsigned char  *W_map_delta2_d;
    unsigned char  *W_map_negative_delta2_d;

    unsigned char  *W_map_delta2_div256_d;
    unsigned char  *W_map_negative_delta2_div256_d;
    unsigned char  *W_map_delta2_div32_d;
    unsigned char  *W_map_negative_delta2_div32_d;

    unsigned char  *W_map_delta_warp_acc_d;
    unsigned char  *W_map_negative_delta_warp_acc_d;

    char  *W_vcsc_values_d;
    unsigned short  *W_vcsc_counts_d;
    unsigned short  *W_vcsc_indices_d;

    unsigned short  *W_vcsc_indices_div16_d;

    unsigned short *W_map_32_div_d;
    unsigned short *W_map_negative_32_div_d;
    unsigned short *W_map_8_div_d;
    unsigned short *W_map_negative_8_div_d;
    unsigned short *W_map_16_div_d;
    unsigned short *W_map_negative_16_div_d;
    unsigned short *W_map_64_div_d;
    unsigned short *W_map_negative_64_div_d;
    unsigned short *W_map_128_div_d;
    unsigned short *W_map_negative_128_div_d;
    unsigned short *W_map_256_div_d;
    unsigned short *W_map_negative_256_div_d;
    unsigned short *W_map_512_div_d;
    unsigned short *W_map_negative_512_div_d;

    if(FLAGS_row_split3_small || FLAGS_row_split_delta2){
        checkCudaErrors(cudaMalloc((void**) &W_map_d, sizeof(unsigned short) * (S / 2) * N));
        checkCudaErrors(cudaMalloc((void**) &W_map_negative_d, sizeof(unsigned short) * (S / 2) * N));
        checkCudaErrors(cudaMalloc((void**) &W_work_d, sizeof(char) * K * N));

        // for 128 split method
        checkCudaErrors(cudaMalloc((void**) &W_map_split_base_d, sizeof(unsigned short) * 128 * N));
        checkCudaErrors(cudaMalloc((void**) &W_map_split_negative_base_d, sizeof(unsigned short) * 128 * N));
        checkCudaErrors(cudaMalloc((void**) &W_map_split_delta_d, sizeof(char) * (S / 2) * N));
        checkCudaErrors(cudaMalloc((void**) &W_map_split_negative_delta_d, sizeof(char) * (S / 2) * N));

        checkCudaErrors(cudaMalloc((void**) &W_map_delta2_d, sizeof(unsigned char) * (S / 2) * N));
        checkCudaErrors(cudaMalloc((void**) &W_map_negative_delta2_d, sizeof(unsigned char) * (S / 2) * N));

        // For 32-divided coalsed access
        checkCudaErrors(cudaMalloc((void**) &W_map_delta2_div256_d, sizeof(unsigned char) * 256 * N * CEIL_DIV( (S/2), 256 ) ));
        checkCudaErrors(cudaMalloc((void**) &W_map_negative_delta2_div256_d, sizeof(unsigned char) * 256 * N * CEIL_DIV( (S/2), 256 ) ));
        checkCudaErrors(cudaMalloc((void**) &W_map_delta2_div32_d, sizeof(unsigned char) * (S / 2) * N));
        checkCudaErrors(cudaMalloc((void**) &W_map_negative_delta2_div32_d, sizeof(unsigned char) * (S / 2) * N));


        checkCudaErrors(cudaMalloc((void**) &W_map_delta_warp_acc_d, sizeof(unsigned char) * (S / 2) * N));
        checkCudaErrors(cudaMalloc((void**) &W_map_negative_delta_warp_acc_d, sizeof(unsigned char) * (S / 2) * N));

        // only Column major
        checkCudaErrors(cudaMalloc((void**) &W_vcsc_values_d, sizeof(char) * 2 * N)); // only -1 & 1
        checkCudaErrors(cudaMalloc((void**) &W_vcsc_counts_d, sizeof(unsigned short) * 2 * N)); // only -1 & 1
        checkCudaErrors(cudaMalloc((void**) &W_vcsc_indices_d, sizeof(unsigned short) * S * N)); // only -1 & 1
        checkCudaErrors(cudaMalloc((void**) &W_vcsc_indices_div16_d, sizeof(unsigned short) * S * N)); // only -1 & 1

        // For 32-divided coalsed access
        checkCudaErrors(cudaMalloc((void**) &W_map_32_div_d, sizeof(unsigned short) * (S / 2) * N));
        checkCudaErrors(cudaMalloc((void**) &W_map_negative_32_div_d, sizeof(unsigned short) * (S / 2) * N));

        // For 8-divided coalsed access
        checkCudaErrors(cudaMalloc((void**) &W_map_8_div_d, sizeof(unsigned short) * (S / 2) * N));
        checkCudaErrors(cudaMalloc((void**) &W_map_negative_8_div_d, sizeof(unsigned short) * (S / 2) * N));

        // For 16-divided coalsed access
        checkCudaErrors(cudaMalloc((void**) &W_map_16_div_d, sizeof(unsigned short) * (S / 2) * N));
        checkCudaErrors(cudaMalloc((void**) &W_map_negative_16_div_d, sizeof(unsigned short) * (S / 2) * N));

        // For 64-divided coalsed access
        checkCudaErrors(cudaMalloc((void**) &W_map_64_div_d, sizeof(unsigned short) * (S / 2) * N));
        checkCudaErrors(cudaMalloc((void**) &W_map_negative_64_div_d, sizeof(unsigned short) * (S / 2) * N));

        // For 128-divided coalsed access
        checkCudaErrors(cudaMalloc((void**) &W_map_128_div_d, sizeof(unsigned short) * (S / 2) * N));
        checkCudaErrors(cudaMalloc((void**) &W_map_negative_128_div_d, sizeof(unsigned short) * (S / 2) * N));

        // For 256-divided coalsed access
        checkCudaErrors(cudaMalloc((void**) &W_map_256_div_d, sizeof(unsigned short) * 256 * N * CEIL_DIV( (S/2), 256 ) ));
        checkCudaErrors(cudaMalloc((void**) &W_map_negative_256_div_d, sizeof(unsigned short) * 256 * N * CEIL_DIV( (S/2), 256 ) ));

        // For 512-divided coalsed access
        checkCudaErrors(cudaMalloc((void**) &W_map_512_div_d, sizeof(unsigned short) * (S / 2) * N));
        checkCudaErrors(cudaMalloc((void**) &W_map_negative_512_div_d, sizeof(unsigned short) * (S / 2) * N));

        prepareW_map<<<N/16, 16>>>(W_work_d, W_map_d, W_map_negative_d,
                                   W_map_split_base_d, W_map_split_negative_base_d,
                                   W_map_split_delta_d, W_map_split_negative_delta_d, W_map_delta2_d, W_map_negative_delta2_d, W_map_delta_warp_acc_d, W_map_negative_delta_warp_acc_d,
                                   W_vcsc_values_d, W_vcsc_counts_d, W_vcsc_indices_d, W_vcsc_indices_div16_d,


                                   W_map_32_div_d, W_map_negative_32_div_d,
                                   W_map_8_div_d, W_map_negative_8_div_d,
                                   W_map_16_div_d, W_map_negative_16_div_d,
                                   W_map_64_div_d, W_map_negative_64_div_d,
                                   W_map_128_div_d, W_map_negative_128_div_d,
                                   W_map_256_div_d, W_map_negative_256_div_d,
                                   W_map_512_div_d, W_map_negative_512_div_d,

                                   W_map_delta2_div256_d, W_map_negative_delta2_div256_d,
                                   W_map_delta2_div32_d, W_map_negative_delta2_div32_d,
                                   ctx_v);
        cudaDeviceSynchronize();
    }


    if(FLAGS_row_split3_small > 0) {
        ms = measureKernel([&]() {
            for (size_t i = 0; i < FLAGS_iter; i++) {
                switch(FLAGS_row_split3_small){

                    case 8:
                        if(M == 1 && K == 6912 && N == 2560 && S == 5120){
                            checkKernelErrors((rowWiseSplit3Small4<1, 6912, 2560, 5120, 8><<< 2560 / 4, dim3(8,4,1) >>>(X_d, W_map_8_div_d,W_map_negative_8_div_d, c_d)));
                        }else if(M == 1 && K == 6912 && N == 2560 && S == 5504){
                            checkKernelErrors((rowWiseSplit3Small4<1, 6912, 2560, 5504, 8><<< 2560 / 4, dim3(8,4,1) >>>(X_d, W_map_8_div_d,W_map_negative_8_div_d, c_d)));
                        }else if(M == 1 && K == 6912 && N == 2560 && S == 4608){
                            checkKernelErrors((rowWiseSplit3Small4<1, 6912, 2560, 4608, 8><<< 2560 / 4, dim3(8,4,1) >>>(X_d, W_map_8_div_d,W_map_negative_8_div_d, c_d)));
                        }else if(M == 1 && K == 6912 && N == 2560 && S == 4160){
                            checkKernelErrors((rowWiseSplit3Small4<1, 6912, 2560, 4160, 8><<< 2560 / 4, dim3(8,4,1) >>>(X_d, W_map_8_div_d,W_map_negative_8_div_d, c_d)));
                        }else if(M == 1 && K == 6912 && N == 2560 && S == 4096){
                            checkKernelErrors((rowWiseSplit3Small4<1, 6912, 2560, 4096, 8><<< 2560 / 4, dim3(8,4,1) >>>(X_d, W_map_8_div_d,W_map_negative_8_div_d, c_d)));
                        }else if(M == 1 && K == 6912 && N == 2560 && S == 3072){
                            checkKernelErrors((rowWiseSplit3Small4<1, 6912, 2560, 3072, 8><<< 2560 / 4, dim3(8,4,1) >>>(X_d, W_map_8_div_d,W_map_negative_8_div_d, c_d)));
                        }else if(M == 1 && K == 6912 && N == 2560 && S == 2752){
                            checkKernelErrors((rowWiseSplit3Small4<1, 6912, 2560, 2752, 8><<< 2560 / 4, dim3(8,4,1) >>>(X_d, W_map_8_div_d,W_map_negative_8_div_d, c_d)));
                        }else if(M == 1 && K == 6912 && N == 2560 && S == 2048){
                            checkKernelErrors((rowWiseSplit3Small4<1, 6912, 2560, 2048, 8><<< 2560 / 4, dim3(8,4,1) >>>(X_d, W_map_8_div_d,W_map_negative_8_div_d, c_d)));
                        }else if(M == 1 && K == 6912 && N == 2560 && S == 1024){
                            checkKernelErrors((rowWiseSplit3Small4<1, 6912, 2560, 1024, 8><<< 2560 / 4, dim3(8,4,1) >>>(X_d, W_map_8_div_d,W_map_negative_8_div_d, c_d)));


                        }else if(M == 8 && K == 6912 && N == 2560 && S == 5504){
                            checkKernelErrors((rowWiseSplit3Small4<8, 6912, 2560, 5504, 8><<< dim3(2560 / 4, 8, 1), dim3(8,4,1) >>>(X_d, W_map_8_div_d,W_map_negative_8_div_d, c_d)));
                        }else if(M == 8 && K == 6912 && N == 2560 && S == 5120){
                            checkKernelErrors((rowWiseSplit3Small4<8, 6912, 2560, 5120, 8><<< dim3(2560 / 4, 8, 1), dim3(8,4,1) >>>(X_d, W_map_8_div_d,W_map_negative_8_div_d, c_d)));
                        }else if(M == 8 && K == 6912 && N == 2560 && S == 4160){
                            checkKernelErrors((rowWiseSplit3Small4<8, 6912, 2560, 4160, 8><<< dim3(2560 / 4, 8, 1), dim3(8,4,1) >>>(X_d, W_map_8_div_d,W_map_negative_8_div_d, c_d)));
                        }else if(M == 8 && K == 6912 && N == 2560 && S == 4096){
                            checkKernelErrors((rowWiseSplit3Small4<8, 6912, 2560, 4096, 8><<< dim3(2560 / 4, 8, 1), dim3(8,4,1) >>>(X_d, W_map_8_div_d,W_map_negative_8_div_d, c_d)));
                        }else if(M == 8 && K == 6912 && N == 2560 && S == 2752){
                            checkKernelErrors((rowWiseSplit3Small4<8, 6912, 2560, 2752, 8><<< dim3(2560 / 4, 8, 1), dim3(8,4,1) >>>(X_d, W_map_8_div_d,W_map_negative_8_div_d, c_d)));
                        }else if(M == 8 && K == 6912 && N == 2560 && S == 2048){
                            checkKernelErrors((rowWiseSplit3Small4<8, 6912, 2560, 2048, 8><<< dim3(2560 / 4, 8, 1), dim3(8,4,1) >>>(X_d, W_map_8_div_d,W_map_negative_8_div_d, c_d)));

                        }else if(M == 4 && K == 2560 && N == 2560 && S == 1024){
                            checkKernelErrors((rowWiseSplit3Small4<4, 2560, 2560, 1024, 8><<< dim3(2560 / 4, 4, 1 ), dim3(8,4,1) >>>(X_d, W_map_8_div_d,W_map_negative_8_div_d, c_d)));
                        }else if(M == 4 && K == 2560 && N == 2560 && S == 1536){
                            checkKernelErrors((rowWiseSplit3Small4<4, 2560, 2560, 1536, 8><<< dim3(2560 / 4, 4, 1 ), dim3(8,4,1) >>>(X_d, W_map_8_div_d,W_map_negative_8_div_d, c_d)));
                        }else if(M == 4 && K == 2560 && N == 2560 && S == 2048){
                            checkKernelErrors((rowWiseSplit3Small4<4, 2560, 2560, 2048, 8><<< dim3(2560 / 4, 4, 1 ), dim3(8,4,1) >>>(X_d, W_map_8_div_d,W_map_negative_8_div_d, c_d)));
                        }else {
                            STRICT_ASSERT(false, "Wrong split size");
                        }
                        break;
                    case 16:
                        if(M == 1 && K == 6912 && N == 2560 && S == 5120){
                            checkKernelErrors((rowWiseSplit3Small4<1, 6912, 2560, 5120, 16><<< 2560 / 2, dim3(16,2,1) >>>(X_d, W_map_16_div_d,W_map_negative_16_div_d, c_d)));
                        }else if(M == 1 && K == 6912 && N == 2560 && S == 4608){
                            checkKernelErrors((rowWiseSplit3Small4<1, 6912, 2560, 4608, 16><<< 2560 / 2, dim3(16,2,1) >>>(X_d, W_map_16_div_d,W_map_negative_16_div_d, c_d)));
                        }else if(M == 1 && K == 6912 && N == 2560 && S == 4096){
                            checkKernelErrors((rowWiseSplit3Small4<1, 6912, 2560, 4096, 16><<< 2560 / 2, dim3(16,2,1) >>>(X_d, W_map_16_div_d,W_map_negative_16_div_d, c_d)));
                        }else if(M == 1 && K == 6912 && N == 2560 && S == 3072){
                            checkKernelErrors((rowWiseSplit3Small4<1, 6912, 2560, 3072, 16><<< 2560 / 2, dim3(16,2,1) >>>(X_d, W_map_16_div_d,W_map_negative_16_div_d, c_d)));
                        }else if(M == 1 && K == 6912 && N == 2560 && S == 2048){
                            checkKernelErrors((rowWiseSplit3Small4<1, 6912, 2560, 2048, 16><<< 2560 / 2, dim3(16,2,1) >>>(X_d, W_map_16_div_d,W_map_negative_16_div_d, c_d)));
                        }else if(M == 1 && K == 6912 && N == 2560 && S == 1024){
                            checkKernelErrors((rowWiseSplit3Small4<1, 6912, 2560, 1024, 16><<< 2560 / 2, dim3(16,2,1) >>>(X_d, W_map_16_div_d,W_map_negative_16_div_d, c_d)));
                        }else {
                            STRICT_ASSERT(false, "Wrong split size");
                        }
                        break;
                    case 32:

                        if(M == 1 && K == 6912 && N == 2560 && S == 5120){
                            checkKernelErrors((rowWiseSplit3Small4<1, 6912, 2560, 5120, 32><<< 2560 / 1, dim3(32,1,1) >>>(X_d, W_map_32_div_d,W_map_negative_32_div_d, c_d)));
                        }else if(M == 1 && K == 6912 && N == 2560 && S == 5504){
                            checkKernelErrors((rowWiseSplit3Small4<1, 6912, 2560, 5504, 32><<< 2560 / 1, dim3(32,1,1) >>>(X_d, W_map_32_div_d,W_map_negative_32_div_d, c_d)));
                        }else if(M == 1 && K == 6912 && N == 2560 && S == 4608){
                            checkKernelErrors((rowWiseSplit3Small4<1, 6912, 2560, 4608, 32><<< 2560 / 1, dim3(32,1,1) >>>(X_d, W_map_32_div_d,W_map_negative_32_div_d, c_d)));
                        }else if(M == 1 && K == 6912 && N == 2560 && S == 4160){
                            checkKernelErrors((rowWiseSplit3Small4<1, 6912, 2560, 4160, 32><<< 2560 / 1, dim3(32,1,1) >>>(X_d, W_map_32_div_d,W_map_negative_32_div_d, c_d)));
                        }else if(M == 1 && K == 6912 && N == 2560 && S == 4096){
                            checkKernelErrors((rowWiseSplit3Small4<1, 6912, 2560, 4096, 32><<< 2560 / 1, dim3(32,1,1) >>>(X_d, W_map_32_div_d,W_map_negative_32_div_d, c_d)));
                        }else if(M == 1 && K == 6912 && N == 2560 && S == 3072){
                            checkKernelErrors((rowWiseSplit3Small4<1, 6912, 2560, 3072, 32><<< 2560 / 1, dim3(32,1,1) >>>(X_d, W_map_32_div_d,W_map_negative_32_div_d, c_d)));
                        }else if(M == 1 && K == 6912 && N == 2560 && S == 2752){
                            checkKernelErrors((rowWiseSplit3Small4<1, 6912, 2560, 2752, 32><<< 2560 / 1, dim3(32,1,1) >>>(X_d, W_map_32_div_d,W_map_negative_32_div_d, c_d)));
                        }else if(M == 1 && K == 6912 && N == 2560 && S == 2048){
                            checkKernelErrors((rowWiseSplit3Small4<1, 6912, 2560, 2048, 32><<< 2560 / 1, dim3(32,1,1) >>>(X_d, W_map_32_div_d,W_map_negative_32_div_d, c_d)));
                        }else if(M == 1 && K == 6912 && N == 2560 && S == 1024) {
                            checkKernelErrors((rowWiseSplit3Small4< 1, 6912, 2560, 1024, 32 ><<< 2560 / 1, dim3(32, 1, 1) >>>(X_d, W_map_32_div_d,W_map_negative_32_div_d, c_d)));

                        }else if(M == 4 && K == 6912 && N == 2560 && S == 5504) {
                            checkKernelErrors((rowWiseSplit3Small4< 4, 6912, 2560, 5504, 32 ><<< dim3(2560, 4, 1), dim3(32, 1, 1) >>>(X_d, W_map_32_div_d,W_map_negative_32_div_d, c_d)));
                        }else if(M == 4 && K == 6912 && N == 2560 && S == 5120) {
                            checkKernelErrors((rowWiseSplit3Small4< 4, 6912, 2560, 5120, 32 ><<< dim3(2560, 4, 1), dim3(32, 1, 1) >>>(X_d, W_map_32_div_d,W_map_negative_32_div_d, c_d)));
                        }else if(M == 4 && K == 6912 && N == 2560 && S == 4096) {
                            checkKernelErrors((rowWiseSplit3Small4 < 4, 6912, 2560, 4096, 32 ><<< dim3(2560, 4, 1), dim3(32, 1, 1) >>>(X_d, W_map_32_div_d,W_map_negative_32_div_d, c_d)));
                        }else if(M == 4 && K == 6912 && N == 2560 && S == 3840) {
                            checkKernelErrors((rowWiseSplit3Small4 < 4, 6912, 2560, 3840, 32 ><<< dim3(2560, 4, 1), dim3(32, 1, 1) >>>(X_d, W_map_32_div_d,W_map_negative_32_div_d, c_d)));
                        }else if(M == 4 && K == 6912 && N == 2560 && S == 2752) {
                            checkKernelErrors((rowWiseSplit3Small4 < 4, 6912, 2560, 2752, 32 ><<< dim3(2560, 4, 1), dim3(32, 1, 1) >>>(X_d, W_map_32_div_d,W_map_negative_32_div_d, c_d)));
                        }else if(M == 4 && K == 6912 && N == 2560 && S == 2048) {
                            checkKernelErrors((rowWiseSplit3Small4 < 4, 6912, 2560, 2048, 32 ><<< dim3(2560, 4, 1), dim3(32, 1, 1) >>>(X_d, W_map_32_div_d,W_map_negative_32_div_d, c_d)));
                        }else if(M == 8 && K == 6912 && N == 2560 && S == 5504) {
                            checkKernelErrors((rowWiseSplit3Small4 < 8, 6912, 2560, 5504, 32 ><<< dim3(2560, 8, 1), dim3(32, 1, 1) >>>(X_d, W_map_32_div_d,W_map_negative_32_div_d, c_d)));

                        }else if(M == 8 && K == 6912 && N == 2560 && S == 5120) {
                            checkKernelErrors((rowWiseSplit3Small4 < 8, 6912, 2560, 5120, 32 ><<< dim3(2560, 8, 1), dim3(32, 1, 1) >>>(X_d, W_map_32_div_d,W_map_negative_32_div_d, c_d)));

                        }else if(M == 8 && K == 6912 && N == 2560 && S == 4096) {
                            checkKernelErrors((rowWiseSplit3Small4 < 8, 6912, 2560, 4096, 32 ><<< dim3(2560, 8, 1), dim3(32, 1, 1) >>>(X_d, W_map_32_div_d,W_map_negative_32_div_d, c_d)));
                        }else if(M == 8 && K == 6912 && N == 2560 && S == 2752) {
                            checkKernelErrors((rowWiseSplit3Small4 < 8, 6912, 2560, 2752, 32 ><<< dim3(2560, 8, 1), dim3(32, 1, 1) >>>(X_d, W_map_32_div_d,W_map_negative_32_div_d, c_d)));
                        }else if(M == 8 && K == 6912 && N == 2560 && S == 2048) {
                            checkKernelErrors((rowWiseSplit3Small4 < 8, 6912, 2560, 2048, 32 ><<< dim3(2560, 8, 1), dim3(32, 1, 1) >>>(X_d, W_map_32_div_d,W_map_negative_32_div_d, c_d)));



                        }else if(M == 4 && K == 2560 && N == 13824 && S == 2048) {
                            checkKernelErrors((rowWiseSplit3Small4 < 4, 2560, 13824, 2048, 32 ><<< dim3(13824, 4, 1), dim3(32, 1, 1) >>>(X_d,W_map_32_div_d,W_map_negative_32_div_d, c_d)));

                        }else if(M == 4 && K == 2560 && N == 13824 && S == 1536) {
                            checkKernelErrors((rowWiseSplit3Small4 < 4, 2560, 13824, 1536, 32 ><<< dim3(13824, 4, 1), dim3(32, 1, 1) >>>(X_d, W_map_32_div_d,W_map_negative_32_div_d, c_d)));

                        }else if(M == 4 && K == 2560 && N == 13824 && S == 1024) {
                            checkKernelErrors((rowWiseSplit3Small4 < 4, 2560, 13824, 1024, 32 ><<< dim3(13824, 4, 1), dim3(32, 1, 1) >>>(X_d, W_map_32_div_d,W_map_negative_32_div_d, c_d)));

                        }else if(M == 8 && K == 2560 && N == 13824 && S == 2048) {
                            checkKernelErrors((rowWiseSplit3Small4 < 8, 2560, 13824, 2048, 32 ><<< dim3(13824, 8, 1), dim3(32, 1, 1) >>>(X_d, W_map_32_div_d,W_map_negative_32_div_d, c_d)));

                        }else if(M == 8 && K == 2560 && N == 13824 && S == 1536) {
                            checkKernelErrors((rowWiseSplit3Small4 < 8, 2560, 13824, 1536, 32 ><<< dim3(13824, 8, 1), dim3(32, 1, 1) >>>(X_d, W_map_32_div_d,W_map_negative_32_div_d, c_d)));

                        }else if(M == 8 && K == 2560 && N == 13824 && S == 1024) {
                            checkKernelErrors((rowWiseSplit3Small4 < 8, 2560, 13824, 1024, 32 ><<< dim3(13824, 8, 1), dim3(32, 1, 1) >>>(X_d, W_map_32_div_d,W_map_negative_32_div_d, c_d)));


                        }else if(M == 4 && K == 2560 && N == 2560 && S == 2048) {
                            checkKernelErrors((rowWiseSplit3Small4 < 4, 2560, 2560, 2048, 32 ><<< dim3(2560, 4, 1), dim3(32, 1, 1) >>>(X_d, W_map_32_div_d,W_map_negative_32_div_d, c_d)));

                        }else if(M == 4 && K == 2560 && N == 2560 && S == 1536) {
                            checkKernelErrors((rowWiseSplit3Small4 < 4, 2560, 2560, 1536, 32 ><<< dim3(2560, 4, 1), dim3(32, 1, 1) >>>(X_d, W_map_32_div_d,W_map_negative_32_div_d, c_d)));

                        }else if(M == 4 && K == 2560 && N == 2560 && S == 1024) {
                            checkKernelErrors((rowWiseSplit3Small4 < 4, 2560, 2560, 1024, 32 ><<< dim3(2560, 4, 1), dim3(32, 1, 1) >>>(X_d, W_map_32_div_d,W_map_negative_32_div_d, c_d)));


                        }else if(M == 8 && K == 2560 && N == 2560 && S == 2048) {
                            checkKernelErrors((rowWiseSplit3Small4 < 8, 2560, 2560, 2048, 32 ><<< dim3(2560, 8, 1), dim3(32, 1, 1) >>>(X_d, W_map_32_div_d,W_map_negative_32_div_d, c_d)));

                        }else if(M == 8 && K == 2560 && N == 2560 && S == 1536) {
                            checkKernelErrors((rowWiseSplit3Small4 < 8, 2560, 2560, 1536, 32 ><<< dim3(2560, 8, 1), dim3(32, 1, 1) >>>(X_d, W_map_32_div_d,W_map_negative_32_div_d, c_d)));

                        }else if(M == 8 && K == 2560 && N == 2560 && S == 1024) {
                            checkKernelErrors((rowWiseSplit3Small4 < 8, 2560, 2560, 1024, 32 ><<< dim3(2560, 8, 1), dim3(32, 1, 1) >>>(X_d, W_map_32_div_d,W_map_negative_32_div_d, c_d)));



                        }else if(M == 4 && K == 2560 && N == 3840 && S == 2048) {
                            checkKernelErrors((rowWiseSplit3Small4 < 4, 2560, 3840, 2048, 32 ><<< dim3(3840, 4, 1), dim3(32, 1, 1) >>>(X_d, W_map_32_div_d,W_map_negative_32_div_d, c_d)));

                        }else if(M == 4 && K == 2560 && N == 3840 && S == 1536) {
                            checkKernelErrors((rowWiseSplit3Small4 < 4, 2560, 3840, 1536, 32 ><<< dim3(3840, 4, 1), dim3(32, 1, 1) >>>(X_d, W_map_32_div_d,W_map_negative_32_div_d, c_d)));

                        }else if(M == 4 && K == 2560 && N == 3840 && S == 1024) {
                            checkKernelErrors((rowWiseSplit3Small4 < 4, 2560, 3840, 1024, 32 ><<< dim3(3840, 4, 1), dim3(32, 1, 1) >>>(X_d, W_map_32_div_d,W_map_negative_32_div_d, c_d)));


                        }else if(M == 8 && K == 2560 && N == 3840 && S == 2048) {
                            checkKernelErrors((rowWiseSplit3Small4 < 8, 2560, 3840, 2048, 32 ><<< dim3(3840, 8, 1), dim3(32, 1, 1) >>>(X_d, W_map_32_div_d,W_map_negative_32_div_d, c_d)));

                        }else if(M == 8 && K == 2560 && N == 3840 && S == 1536) {
                            checkKernelErrors((rowWiseSplit3Small4 < 8, 2560, 3840, 1536, 32 ><<< dim3(3840, 8, 1), dim3(32, 1, 1) >>>(X_d, W_map_32_div_d,W_map_negative_32_div_d, c_d)));

                        }else if(M == 8 && K == 2560 && N == 3840 && S == 1024) {
                            checkKernelErrors((rowWiseSplit3Small4 < 8, 2560, 3840, 1024, 32 ><<< dim3(3840, 8, 1), dim3(32, 1, 1) >>>(X_d, W_map_32_div_d,W_map_negative_32_div_d, c_d)));



                        }else if(M == 1 && K == 2560 && N == 13824 && S == 2048) {
                            checkKernelErrors((rowWiseSplit3Small4 < 1, 2560, 13824, 2048, 32 ><<< dim3(13824, 1, 1), dim3(32, 1, 1) >>>(X_d, W_map_32_div_d,W_map_negative_32_div_d, c_d)));
                        }else if(M == 1 && K == 2560 && N == 13824 && S == 1536) {
                            checkKernelErrors((rowWiseSplit3Small4 < 1, 2560, 13824, 1536, 32 ><<< dim3(13824, 1, 1), dim3(32, 1, 1) >>>(X_d, W_map_32_div_d,W_map_negative_32_div_d, c_d)));
                        }else if(M == 1 && K == 2560 && N == 13824 && S == 1024) {
                            checkKernelErrors((rowWiseSplit3Small4 < 1, 2560, 13824, 1024, 32 ><<< dim3(13824, 1, 1), dim3(32, 1, 1) >>>(X_d, W_map_32_div_d,W_map_negative_32_div_d, c_d)));



                        }else if(M == 1 && K == 2560 && N == 2560 && S == 2048) {
                            checkKernelErrors((rowWiseSplit3Small4 < 1, 2560, 2560, 2048, 32 ><<< dim3(2560, 1, 1), dim3(32, 1, 1) >>>(X_d, W_map_32_div_d,W_map_negative_32_div_d, c_d)));
                        }else if(M == 1 && K == 2560 && N == 2560 && S == 1536) {
                            checkKernelErrors((rowWiseSplit3Small4 < 1, 2560, 2560, 1536, 32 ><<< dim3(2560, 1, 1), dim3(32, 1, 1) >>>(X_d, W_map_32_div_d,W_map_negative_32_div_d, c_d)));
                        }else if(M == 1 && K == 2560 && N == 2560 && S == 1024) {
                            checkKernelErrors((rowWiseSplit3Small4 < 1, 2560, 2560, 1024, 32 ><<< dim3(2560, 1, 1), dim3(32, 1, 1) >>>(X_d, W_map_32_div_d,W_map_negative_32_div_d, c_d)));


                        }else if(M == 1 && K == 2560 && N == 3840 && S == 2048) {
                            checkKernelErrors((rowWiseSplit3Small4 < 1, 2560, 3840, 2048, 32 ><<< dim3(3840, 1, 1), dim3(32, 1, 1) >>>(X_d, W_map_32_div_d,W_map_negative_32_div_d, c_d)));
                        }else if(M == 1 && K == 2560 && N == 3840 && S == 1536) {
                            checkKernelErrors((rowWiseSplit3Small4 < 1, 2560, 3840, 1536, 32 ><<< dim3(3840, 1, 1), dim3(32, 1, 1) >>>(X_d, W_map_32_div_d,W_map_negative_32_div_d, c_d)));
                        }else if(M == 1 && K == 2560 && N == 3840 && S == 1024) {
                            checkKernelErrors((rowWiseSplit3Small4 < 1, 2560, 3840, 1024, 32 ><<< dim3(3840, 1, 1), dim3(32, 1, 1) >>>(X_d, W_map_32_div_d,W_map_negative_32_div_d, c_d)));


                            // For 8B model
                        }else if(M == 1 && K == 14336 && N == 4096) {
                            if(S == 8960){ // 40
                                checkKernelErrors((rowWiseSplit3Small4 < 1, 14336, 4096, 8960, 32 ><<< dim3(4096, 1, 1), dim3(32, 1, 1) >>>(X_d, W_map_32_div_d,W_map_negative_32_div_d, c_d)));
                                break;
                            }
                            if(S == 8704){ // 40
                                checkKernelErrors((rowWiseSplit3Small4 < 1, 14336, 4096, 8704, 32 ><<< dim3(4096, 1, 1), dim3(32, 1, 1) >>>(X_d, W_map_32_div_d,W_map_negative_32_div_d, c_d)));
                                break;
                            }
                            if(S == 8576){ // 40
                                checkKernelErrors((rowWiseSplit3Small4 < 1, 14336, 4096, 8576, 32 ><<< dim3(4096, 1, 1), dim3(32, 1, 1) >>>(X_d, W_map_32_div_d,W_map_negative_32_div_d, c_d)));
                                break;
                            }
                            if(S == 5760){ // 60
                                checkKernelErrors((rowWiseSplit3Small4 < 1, 14336, 4096, 5760, 32 ><<< dim3(4096, 1, 1), dim3(32, 1, 1) >>>(X_d, W_map_32_div_d,W_map_negative_32_div_d, c_d)));
                                break;
                            }
                            if(S == 5632){ // 60
                                checkKernelErrors((rowWiseSplit3Small4 < 1, 14336, 4096, 5632, 32 ><<< dim3(4096, 1, 1), dim3(32, 1, 1) >>>(X_d, W_map_32_div_d,W_map_negative_32_div_d, c_d)));
                                break;
                            }
                            if(S == 5376){ // 60
                                checkKernelErrors((rowWiseSplit3Small4 < 1, 14336, 4096, 5376, 32 ><<< dim3(4096, 1, 1), dim3(32, 1, 1) >>>(X_d, W_map_32_div_d,W_map_negative_32_div_d, c_d)));
                                break;
                            }
                            STRICT_ASSERT(false, "Wrong split size");

                        }else if(M == 1 && K == 4096 && N == 28672) {
                            if(S == 2560){ // 40
                                checkKernelErrors((rowWiseSplit3Small4 < 1, 4096, 28672, 2560, 32 ><<< dim3(28672, 1, 1), dim3(32, 1, 1) >>>(X_d, W_map_32_div_d,W_map_negative_32_div_d, c_d)));
                                break;
                            }
                            if(S == 2496){ // 40
                                checkKernelErrors((rowWiseSplit3Small4 < 1, 4096, 28672, 2496, 32 ><<< dim3(28672, 1, 1), dim3(32, 1, 1) >>>(X_d, W_map_32_div_d,W_map_negative_32_div_d, c_d)));
                                break;
                            }
                            if(S == 1600){ // 60
                                checkKernelErrors((rowWiseSplit3Small4 < 1, 4096, 28672, 1600, 32 ><<< dim3(28672, 1, 1), dim3(32, 1, 1) >>>(X_d, W_map_32_div_d,W_map_negative_32_div_d, c_d)));
                                break;
                            }
                            if(S == 1536){ // 60
                                checkKernelErrors((rowWiseSplit3Small4 < 1, 4096, 28672, 1536, 32 ><<< dim3(28672, 1, 1), dim3(32, 1, 1) >>>(X_d, W_map_32_div_d,W_map_negative_32_div_d, c_d)));
                                break;
                            }
                            STRICT_ASSERT(false, "Wrong split size");

                        }else if(M == 1 && K == 4096 && N == 4096) {
                            if(S == 2560){ // 40
                                checkKernelErrors((rowWiseSplit3Small4 < 1, 4096, 4096, 2560, 32 ><<< dim3(4096, 1, 1), dim3(32, 1, 1) >>>(X_d, W_map_32_div_d,W_map_negative_32_div_d, c_d)));
                                break;
                            }
                            if(S == 2496){ // 40
                                checkKernelErrors((rowWiseSplit3Small4 < 1, 4096, 4096, 2496, 32 ><<< dim3(4096, 1, 1), dim3(32, 1, 1) >>>(X_d, W_map_32_div_d,W_map_negative_32_div_d, c_d)));
                                break;
                            }
                            if(S == 1600){ // 60
                                checkKernelErrors((rowWiseSplit3Small4 < 1, 4096, 4096, 1600, 32 ><<< dim3(4096, 1, 1), dim3(32, 1, 1) >>>(X_d, W_map_32_div_d,W_map_negative_32_div_d, c_d)));
                                break;
                            }
                            if(S == 1536){ // 60
                                checkKernelErrors((rowWiseSplit3Small4 < 1, 4096, 4096, 1536, 32 ><<< dim3(4096, 1, 1), dim3(32, 1, 1) >>>(X_d, W_map_32_div_d,W_map_negative_32_div_d, c_d)));
                                break;
                            }
                            STRICT_ASSERT(false, "Wrong split size");

                        }else if(M == 1 && K == 4096 && N == 6144) {
                            if(S == 2560){ // 40
                                checkKernelErrors((rowWiseSplit3Small4 < 1, 4096, 6144, 2560, 32 ><<< dim3(6144, 1, 1), dim3(32, 1, 1) >>>(X_d, W_map_32_div_d,W_map_negative_32_div_d, c_d)));
                                break;
                            }
                            if(S == 2496){ // 40
                                checkKernelErrors((rowWiseSplit3Small4 < 1, 4096, 6144, 2496, 32 ><<< dim3(6144, 1, 1), dim3(32, 1, 1) >>>(X_d, W_map_32_div_d,W_map_negative_32_div_d, c_d)));
                                break;
                            }
                            if(S == 1600){ // 60
                                checkKernelErrors((rowWiseSplit3Small4 < 1, 4096, 6144, 1600, 32 ><<< dim3(6144, 1, 1), dim3(32, 1, 1) >>>(X_d, W_map_32_div_d,W_map_negative_32_div_d, c_d)));
                                break;
                            }
                            if(S == 1536){ // 60
                                checkKernelErrors((rowWiseSplit3Small4 < 1, 4096, 6144, 1536, 32 ><<< dim3(6144, 1, 1), dim3(32, 1, 1) >>>(X_d, W_map_32_div_d,W_map_negative_32_div_d, c_d)));
                                break;
                            }
                            STRICT_ASSERT(false, "Wrong split size");



                        }else if(M == 4 && K == 14336 && N == 4096) {
                            if(S == 8960){ // 40
                                checkKernelErrors((rowWiseSplit3Small4 < 4, 14336, 4096, 8960, 32 ><<< dim3(4096, 4, 1), dim3(32, 1, 1) >>>(X_d, W_map_32_div_d,W_map_negative_32_div_d, c_d)));
                                break;
                            }
                            if(S == 8704){ // 40
                                checkKernelErrors((rowWiseSplit3Small4 < 4, 14336, 4096, 8704, 32 ><<< dim3(4096, 4, 1), dim3(32, 1, 1) >>>(X_d, W_map_32_div_d,W_map_negative_32_div_d, c_d)));
                                break;
                            }
                            if(S == 8576){ // 40
                                checkKernelErrors((rowWiseSplit3Small4 < 4, 14336, 4096, 8576, 32 ><<< dim3(4096, 4, 1), dim3(32, 1, 1) >>>(X_d, W_map_32_div_d,W_map_negative_32_div_d, c_d)));
                                break;
                            }
                            if(S == 5760){ // 60
                                checkKernelErrors((rowWiseSplit3Small4 < 4, 14336, 4096, 5760, 32 ><<< dim3(4096, 4, 1), dim3(32, 1, 1) >>>(X_d, W_map_32_div_d,W_map_negative_32_div_d, c_d)));
                                break;
                            }
                            if(S == 5632){ // 60
                                checkKernelErrors((rowWiseSplit3Small4 < 4, 14336, 4096, 5632, 32 ><<< dim3(4096, 4, 1), dim3(32, 1, 1) >>>(X_d, W_map_32_div_d,W_map_negative_32_div_d, c_d)));
                                break;
                            }
                            if(S == 5376){ // 60
                                checkKernelErrors((rowWiseSplit3Small4 < 4, 14336, 4096, 5376, 32 ><<< dim3(4096, 4, 1), dim3(32, 1, 1) >>>(X_d, W_map_32_div_d,W_map_negative_32_div_d, c_d)));
                                break;
                            }
                            STRICT_ASSERT(false, "Wrong split size");

                        }else if(M == 4 && K == 4096 && N == 28672) {
                            if(S == 2560){ // 40
                                checkKernelErrors((rowWiseSplit3Small4 < 4, 4096, 28672, 2560, 32 ><<< dim3(28672, 4, 1), dim3(32, 1, 1) >>>(X_d, W_map_32_div_d,W_map_negative_32_div_d, c_d)));
                                break;
                            }
                            if(S == 2496){ // 40
                                checkKernelErrors((rowWiseSplit3Small4 < 4, 4096, 28672, 2496, 32 ><<< dim3(28672, 4, 1), dim3(32, 1, 1) >>>(X_d, W_map_32_div_d,W_map_negative_32_div_d, c_d)));
                                break;
                            }
                            if(S == 1600){ // 60
                                checkKernelErrors((rowWiseSplit3Small4 < 4, 4096, 28672, 1600, 32 ><<< dim3(28672, 4, 1), dim3(32, 1, 1) >>>(X_d, W_map_32_div_d,W_map_negative_32_div_d, c_d)));
                                break;
                            }
                            if(S == 1536){ // 60
                                checkKernelErrors((rowWiseSplit3Small4 < 4, 4096, 28672, 1536, 32 ><<< dim3(28672, 4, 1), dim3(32, 1, 1) >>>(X_d, W_map_32_div_d,W_map_negative_32_div_d, c_d)));
                                break;
                            }
                            STRICT_ASSERT(false, "Wrong split size");

                        }else if(M == 4 && K == 4096 && N == 4096) {
                            if(S == 2560){ // 40
                                checkKernelErrors((rowWiseSplit3Small4 < 4, 4096, 4096, 2560, 32 ><<< dim3(4096, 4, 1), dim3(32, 1, 1) >>>(X_d, W_map_32_div_d,W_map_negative_32_div_d, c_d)));
                                break;
                            }
                            if(S == 2496){ // 40
                                checkKernelErrors((rowWiseSplit3Small4 < 4, 4096, 4096, 2496, 32 ><<< dim3(4096, 4, 1), dim3(32, 1, 1) >>>(X_d, W_map_32_div_d,W_map_negative_32_div_d, c_d)));
                                break;
                            }
                            if(S == 1600){ // 60
                                checkKernelErrors((rowWiseSplit3Small4 < 4, 4096, 4096, 1600, 32 ><<< dim3(4096, 4, 1), dim3(32, 1, 1) >>>(X_d, W_map_32_div_d,W_map_negative_32_div_d, c_d)));
                                break;
                            }
                            if(S == 1536){ // 60
                                checkKernelErrors((rowWiseSplit3Small4 < 4, 4096, 4096, 1536, 32 ><<< dim3(4096, 4, 1), dim3(32, 1, 1) >>>(X_d, W_map_32_div_d,W_map_negative_32_div_d, c_d)));
                                break;
                            }
                            STRICT_ASSERT(false, "Wrong split size");

                        }else if(M == 4 && K == 4096 && N == 6144) {
                            if(S == 2560){ // 40
                                checkKernelErrors((rowWiseSplit3Small4 < 4, 4096, 6144, 2560, 32 ><<< dim3(6144, 4, 1), dim3(32, 1, 1) >>>(X_d, W_map_32_div_d,W_map_negative_32_div_d, c_d)));
                                break;
                            }
                            if(S == 2496){ // 40
                                checkKernelErrors((rowWiseSplit3Small4 < 4, 4096, 6144, 2496, 32 ><<< dim3(6144, 4, 1), dim3(32, 1, 1) >>>(X_d, W_map_32_div_d,W_map_negative_32_div_d, c_d)));
                                break;
                            }
                            if(S == 1600){ // 60
                                checkKernelErrors((rowWiseSplit3Small4 < 4, 4096, 6144, 1600, 32 ><<< dim3(6144, 4, 1), dim3(32, 1, 1) >>>(X_d, W_map_32_div_d,W_map_negative_32_div_d, c_d)));
                                break;
                            }
                            if(S == 1536){ // 60
                                checkKernelErrors((rowWiseSplit3Small4 < 4, 4096, 6144, 1536, 32 ><<< dim3(6144, 4, 1), dim3(32, 1, 1) >>>(X_d, W_map_32_div_d,W_map_negative_32_div_d, c_d)));
                                break;
                            }
                            STRICT_ASSERT(false, "Wrong split size");



                        }else if(M == 8 && K == 14336 && N == 4096) {
                            if(S == 8960){ // 40
                                checkKernelErrors((rowWiseSplit3Small4 < 8, 14336, 4096, 8960, 32 ><<< dim3(4096, 8, 1), dim3(32, 1, 1) >>>(X_d, W_map_32_div_d,W_map_negative_32_div_d, c_d)));
                                break;
                            }
                            if(S == 8704){ // 40
                                checkKernelErrors((rowWiseSplit3Small4 < 8, 14336, 4096, 8704, 32 ><<< dim3(4096, 8, 1), dim3(32, 1, 1) >>>(X_d, W_map_32_div_d,W_map_negative_32_div_d, c_d)));
                                break;
                            }
                            if(S == 8576){ // 40
                                checkKernelErrors((rowWiseSplit3Small4 < 8, 14336, 4096, 8576, 32 ><<< dim3(4096, 8, 1), dim3(32, 1, 1) >>>(X_d, W_map_32_div_d,W_map_negative_32_div_d, c_d)));
                                break;
                            }
                            if(S == 5760){ // 60
                                checkKernelErrors((rowWiseSplit3Small4 < 8, 14336, 4096, 5760, 32 ><<< dim3(4096, 8, 1), dim3(32, 1, 1) >>>(X_d, W_map_32_div_d,W_map_negative_32_div_d, c_d)));
                                break;
                            }
                            if(S == 5632){ // 60
                                checkKernelErrors((rowWiseSplit3Small4 < 8, 14336, 4096, 5632, 32 ><<< dim3(4096, 8, 1), dim3(32, 1, 1) >>>(X_d, W_map_32_div_d,W_map_negative_32_div_d, c_d)));
                                break;
                            }
                            if(S == 5376){ // 60
                                checkKernelErrors((rowWiseSplit3Small4 < 8, 14336, 4096, 5376, 32 ><<< dim3(4096, 8, 1), dim3(32, 1, 1) >>>(X_d, W_map_32_div_d,W_map_negative_32_div_d, c_d)));
                                break;
                            }
                            STRICT_ASSERT(false, "Wrong split size");

                        }else if(M == 8 && K == 4096 && N == 28672) {
                            if(S == 2560){ // 40
                                checkKernelErrors((rowWiseSplit3Small4 < 8, 4096, 28672, 2560, 32 ><<< dim3(28672, 8, 1), dim3(32, 1, 1) >>>(X_d, W_map_32_div_d,W_map_negative_32_div_d, c_d)));
                                break;
                            }
                            if(S == 2496){ // 40
                                checkKernelErrors((rowWiseSplit3Small4 < 8, 4096, 28672, 2496, 32 ><<< dim3(28672, 8, 1), dim3(32, 1, 1) >>>(X_d, W_map_32_div_d,W_map_negative_32_div_d, c_d)));
                                break;
                            }
                            if(S == 1600){ // 60
                                checkKernelErrors((rowWiseSplit3Small4 < 8, 4096, 28672, 1600, 32 ><<< dim3(28672, 8, 1), dim3(32, 1, 1) >>>(X_d, W_map_32_div_d,W_map_negative_32_div_d, c_d)));
                                break;
                            }
                            if(S == 1536){ // 60
                                checkKernelErrors((rowWiseSplit3Small4 < 8, 4096, 28672, 1536, 32 ><<< dim3(28672, 8, 1), dim3(32, 1, 1) >>>(X_d, W_map_32_div_d,W_map_negative_32_div_d, c_d)));
                                break;
                            }
                            STRICT_ASSERT(false, "Wrong split size");

                        }else if(M == 8 && K == 4096 && N == 4096) {
                            if(S == 2560){ // 40
                                checkKernelErrors((rowWiseSplit3Small4 < 8, 4096, 4096, 2560, 32 ><<< dim3(4096, 8, 1), dim3(32, 1, 1) >>>(X_d, W_map_32_div_d,W_map_negative_32_div_d, c_d)));
                                break;
                            }
                            if(S == 2496){ // 40
                                checkKernelErrors((rowWiseSplit3Small4 < 8, 4096, 4096, 2496, 32 ><<< dim3(4096, 8, 1), dim3(32, 1, 1) >>>(X_d, W_map_32_div_d,W_map_negative_32_div_d, c_d)));
                                break;
                            }
                            if(S == 1600){ // 60
                                checkKernelErrors((rowWiseSplit3Small4 < 8, 4096, 4096, 1600, 32 ><<< dim3(4096, 8, 1), dim3(32, 1, 1) >>>(X_d, W_map_32_div_d,W_map_negative_32_div_d, c_d)));
                                break;
                            }
                            if(S == 1536){ // 60
                                checkKernelErrors((rowWiseSplit3Small4 < 8, 4096, 4096, 1536, 32 ><<< dim3(4096, 8, 1), dim3(32, 1, 1) >>>(X_d, W_map_32_div_d,W_map_negative_32_div_d, c_d)));
                                break;
                            }
                            STRICT_ASSERT(false, "Wrong split size");

                        }else if(M == 8 && K == 4096 && N == 6144) {
                            if(S == 2560){ // 40
                                checkKernelErrors((rowWiseSplit3Small4 < 8, 4096, 6144, 2560, 32 ><<< dim3(6144, 8, 1), dim3(32, 1, 1) >>>(X_d, W_map_32_div_d,W_map_negative_32_div_d, c_d)));
                                break;
                            }
                            if(S == 2496){ // 40
                                checkKernelErrors((rowWiseSplit3Small4 < 8, 4096, 6144, 2496, 32 ><<< dim3(6144, 8, 1), dim3(32, 1, 1) >>>(X_d, W_map_32_div_d,W_map_negative_32_div_d, c_d)));
                                break;
                            }
                            if(S == 1600){ // 60
                                checkKernelErrors((rowWiseSplit3Small4 < 8, 4096, 6144, 1600, 32 ><<< dim3(6144, 8, 1), dim3(32, 1, 1) >>>(X_d, W_map_32_div_d,W_map_negative_32_div_d, c_d)));
                                break;
                            }
                            if(S == 1536){ // 60
                                checkKernelErrors((rowWiseSplit3Small4 < 8, 4096, 6144, 1536, 32 ><<< dim3(6144, 8, 1), dim3(32, 1, 1) >>>(X_d, W_map_32_div_d,W_map_negative_32_div_d, c_d)));
                                break;
                            }
                            STRICT_ASSERT(false, "Wrong split size");


                        }else {
                            STRICT_ASSERT(false, "Wrong split size");
                        }
                        break;
                }
                checkCudaErrors(cudaDeviceSynchronize());
            }
        });
        std::cout << ms / ((float) FLAGS_iter) << std::endl;
    }


    if(FLAGS_row_split_delta2 > 0) {
        ms = measureKernel([&]() {
            for (size_t i = 0; i < FLAGS_iter; i++) {
                switch(FLAGS_row_split_delta2){
                    case 32: {
                        if(M == 1 && K == 6912 && N == 2560){
                            if(S == 5504){
                                //checkKernelErrors((rowWiseSplit2DeltaSmall<1, 6912, 2560, 5504, 32><<< 2560, 32 >>>(X_d, W_map_delta2_d,W_map_negative_delta2_d, c_d)));
                                //checkKernelErrors((rowWiseSplit2DeltaSmall2<1, 6912, 2560, 5504, 32><<< 2560, 32 >>>(X_d, W_map_delta2_div32_d,W_map_negative_delta2_div32_d, c_d)));
                                checkKernelErrors((rowWiseSplit2DeltaSmallAsync3<1, 6912, 2560, 5504, 32><<< 2560, 32 >>>(X_d, W_map_delta2_div256_d,W_map_negative_delta2_div256_d, c_d)));
                                break;
                            }
                            if(S == 5120){
                                checkKernelErrors((rowWiseSplit2DeltaSmallAsync3<1, 6912, 2560, 5120, 32><<< 2560, 32 >>>(X_d, W_map_delta2_div256_d,W_map_negative_delta2_div256_d, c_d)));
                                break;
                            }
                            if(S == 4608){
                                //checkKernelErrors((rowWiseSplit2DeltaSmall<1, 6912, 2560, 4608, 32><<< 2560, 32 >>>(X_d, W_map_delta2_d,W_map_negative_delta2_d, c_d)));
                                checkKernelErrors((rowWiseSplit2DeltaSmallAsync2<1, 6912, 2560, 4608, 32><<< 2560, 32 >>>(X_d, W_map_delta2_div256_d,W_map_negative_delta2_div256_d, c_d)));
                                break;
                            }
                            // no 4160!
                            if(S == 4096){
                                checkKernelErrors((rowWiseSplit2DeltaSmallAsync2<1, 6912, 2560, 4096, 32><<< 2560, 32 >>>(X_d, W_map_delta2_div256_d,W_map_negative_delta2_div256_d, c_d)));
                                //checkKernelErrors((rowWiseSplit2DeltaSmall2<1, 6912, 2560, 4096, 32><<< 2560, 32 >>>(X_d, W_map_delta2_div32_d,W_map_negative_delta2_div32_d, c_d)));
                                //checkKernelErrors((rowWiseSplit2DeltaSmall<1, 6912, 2560, 4096, 32><<< 2560, 32 >>>(X_d, W_map_delta2_d,W_map_negative_delta2_d, c_d)));
                                break;
                            }
                            // no 2752!
                            if(S == 2752){
                                //checkKernelErrors((rowWiseSplit2DeltaSmall<1, 6912, 2560, 2752, 32><<< 2560, 32 >>>(X_d, W_map_delta2_d,W_map_negative_delta2_d, c_d)));
                                checkKernelErrors((rowWiseSplit2DeltaSmallAsync3<1, 6912, 2560, 2752, 32><<< 2560, 32 >>>(X_d, W_map_delta2_div256_d,W_map_negative_delta2_div256_d, c_d)));
                                break;
                            }
                            if(S == 2048){
                                //checkKernelErrors((rowWiseSplit2DeltaSmall<1, 6912, 2560, 2048, 32><<< 2560, 32 >>>(X_d, W_map_delta2_d,W_map_negative_delta2_d, c_d)));
                                checkKernelErrors((rowWiseSplit2DeltaSmallAsync2<1, 6912, 2560, 2048, 32><<< 2560, 32 >>>(X_d, W_map_delta2_div256_d,W_map_negative_delta2_div256_d, c_d)));
                                break;
                            }
                        }
                        if(M == 4 && K == 6912 && N == 2560){
                            if(S == 5504){
                                checkKernelErrors((rowWiseSplit2DeltaSmallAsync3<4, 6912, 2560, 5504, 32><<< dim3(2560, 4, 1), 32 >>>(X_d, W_map_delta2_div256_d,W_map_negative_delta2_div256_d, c_d)));
                                break;
                            }
                            if(S == 5120){
                                checkKernelErrors((rowWiseSplit2DeltaSmallAsync2<4, 6912, 2560, 5120, 32><<< dim3(2560, 4, 1), 32 >>>(X_d, W_map_delta2_div256_d,W_map_negative_delta2_div256_d, c_d)));
                                break;
                            }
                            if(S == 4096){
                                checkKernelErrors((rowWiseSplit2DeltaSmallAsync2<4, 6912, 2560, 4096, 32><<< dim3(2560, 4, 1), 32 >>>(X_d, W_map_delta2_div256_d,W_map_negative_delta2_div256_d, c_d)));
                                break;
                            }
                            if(S == 3456){
                                checkKernelErrors((rowWiseSplit2DeltaSmallAsync3<4, 6912, 2560, 3456, 32><<< dim3(2560, 4, 1), 32 >>>(X_d, W_map_delta2_div256_d,W_map_negative_delta2_div256_d, c_d)));
                                break;
                            }
                            if(S == 2752){
                                checkKernelErrors((rowWiseSplit2DeltaSmallAsync3<4, 6912, 2560, 2752, 32><<< dim3(2560, 4, 1), 32 >>>(X_d, W_map_delta2_div256_d,W_map_negative_delta2_div256_d, c_d)));
                                break;
                            }
                            if(S == 2048){
                                checkKernelErrors((rowWiseSplit2DeltaSmallAsync2<4, 6912, 2560, 2048, 32><<< dim3(2560, 4, 1), 32 >>>(X_d, W_map_delta2_div256_d,W_map_negative_delta2_div256_d, c_d)));
                                break;
                            }
                        }
                        if(M == 8 && K == 6912 && N == 2560){
                            if(S == 5504){
                                checkKernelErrors((rowWiseSplit2DeltaSmallAsync3<8, 6912, 2560, 5504, 32><<< dim3(2560, 8, 1), 32 >>>(X_d, W_map_delta2_div256_d,W_map_negative_delta2_div256_d, c_d)));
                                break;
                            }
                            if(S == 5120){
                                checkKernelErrors((rowWiseSplit2DeltaSmallAsync2<8, 6912, 2560, 5120, 32><<< dim3(2560, 8, 1), 32 >>>(X_d, W_map_delta2_div256_d,W_map_negative_delta2_div256_d, c_d)));
                                break;
                            }
                            if(S == 4096){
                                checkKernelErrors((rowWiseSplit2DeltaSmallAsync2<8, 6912, 2560, 4096, 32><<< dim3(2560, 8, 1), 32 >>>(X_d, W_map_delta2_div256_d,W_map_negative_delta2_div256_d, c_d)));
                                break;
                            }
                            if(S == 2752){
                                checkKernelErrors((rowWiseSplit2DeltaSmallAsync3<8, 6912, 2560, 2752, 32><<< dim3(2560, 8, 1), 32 >>>(X_d, W_map_delta2_div256_d,W_map_negative_delta2_div256_d, c_d)));
                                break;
                            }
                            if(S == 2048){
                                checkKernelErrors((rowWiseSplit2DeltaSmallAsync2<8, 6912, 2560, 2048, 32><<< dim3(2560, 8, 1), 32 >>>(X_d, W_map_delta2_div256_d,W_map_negative_delta2_div256_d, c_d)));
                                break;
                            }
                        }
                        if(M == 1 && K == 2560 && N == 13824){
                            if(S == 2048){
                                checkKernelErrors((rowWiseSplit2DeltaSmallAsync2<1, 2560, 13824, 2048, 32><<< 13824, 32 >>>(X_d, W_map_delta2_div256_d,W_map_negative_delta2_div256_d, c_d)));
                                break;
                            }
                            if(S == 1536){
                                checkKernelErrors((rowWiseSplit2DeltaSmallAsync2<1, 2560, 13824, 1536, 32><<< 13824, 32 >>>(X_d, W_map_delta2_div256_d,W_map_negative_delta2_div256_d, c_d)));
                                break;
                            }
                            if(S == 1024){
                                checkKernelErrors((rowWiseSplit2DeltaSmallAsync2<1, 2560, 13824, 1024, 32><<< 13824, 32 >>>(X_d, W_map_delta2_div256_d,W_map_negative_delta2_div256_d, c_d)));
                                break;
                            }
                        }
                        if(M == 4 && K == 2560 && N == 13824){
                            if(S == 2048){
                                checkKernelErrors((rowWiseSplit2DeltaSmallAsync2<4, 2560, 13824, 2048, 32><<< dim3(13824, 4, 1), 32 >>>(X_d, W_map_delta2_div256_d,W_map_negative_delta2_div256_d, c_d)));
                                break;
                            }
                            if(S == 1536){
                                checkKernelErrors((rowWiseSplit2DeltaSmallAsync2<4, 2560, 13824, 1536, 32><<< dim3(13824, 4, 1), 32 >>>(X_d, W_map_delta2_div256_d,W_map_negative_delta2_div256_d, c_d)));
                                break;
                            }
                            if(S == 1024){
                                checkKernelErrors((rowWiseSplit2DeltaSmallAsync2<4, 2560, 13824, 1024, 32><<< dim3(13824, 4, 1), 32 >>>(X_d, W_map_delta2_div256_d,W_map_negative_delta2_div256_d, c_d)));
                                break;
                            }
                        }
                        if(M == 8 && K == 2560 && N == 13824){
                            if(S == 2048){
                                checkKernelErrors((rowWiseSplit2DeltaSmallAsync2<8, 2560, 13824, 2048, 32><<< dim3(13824, 8, 1), 32 >>>(X_d, W_map_delta2_div256_d,W_map_negative_delta2_div256_d, c_d)));
                                break;
                            }
                            if(S == 1536){
                                checkKernelErrors((rowWiseSplit2DeltaSmallAsync2<8, 2560, 13824, 1536, 32><<< dim3(13824, 8, 1), 32 >>>(X_d, W_map_delta2_div256_d,W_map_negative_delta2_div256_d, c_d)));
                                break;
                            }
                            if(S == 1024){
                                checkKernelErrors((rowWiseSplit2DeltaSmallAsync2<8, 2560, 13824, 1024, 32><<< dim3(13824, 8, 1), 32 >>>(X_d, W_map_delta2_div256_d,W_map_negative_delta2_div256_d, c_d)));
                                break;
                            }
                        }
                        if(M == 1 && K == 2560 && N == 2560){
                            if(S == 2048){
                                checkKernelErrors((rowWiseSplit2DeltaSmallAsync2<1, 2560, 2560, 2048, 32><<< dim3(2560, 1, 1), 32 >>>(X_d, W_map_delta2_div256_d,W_map_negative_delta2_div256_d, c_d)));
                                break;
                            }
                            if(S == 1536){
                                checkKernelErrors((rowWiseSplit2DeltaSmallAsync2<1, 2560, 2560, 1536, 32><<< dim3(2560, 1, 1), 32 >>>(X_d, W_map_delta2_div256_d,W_map_negative_delta2_div256_d, c_d)));
                                break;
                            }
                            if(S == 1024){
                                checkKernelErrors((rowWiseSplit2DeltaSmallAsync2<1, 2560, 2560, 1024, 32><<< dim3(2560, 1, 1), 32 >>>(X_d, W_map_delta2_div256_d,W_map_negative_delta2_div256_d, c_d)));
                                break;
                            }
                        }
                        if(M == 4 && K == 2560 && N == 2560){
                            if(S == 2048){
                                checkKernelErrors((rowWiseSplit2DeltaSmallAsync2<4, 2560, 2560, 2048, 32><<< dim3(2560, 4, 1), 32 >>>(X_d, W_map_delta2_div256_d,W_map_negative_delta2_div256_d, c_d)));
                                break;
                            }
                            if(S == 1536){
                                checkKernelErrors((rowWiseSplit2DeltaSmallAsync2<4, 2560, 2560, 1536, 32><<< dim3(2560, 4, 1), 32 >>>(X_d, W_map_delta2_div256_d,W_map_negative_delta2_div256_d, c_d)));
                                break;
                            }
                            if(S == 1024){
                                checkKernelErrors((rowWiseSplit2DeltaSmallAsync2<4, 2560, 2560, 1024, 32><<< dim3(2560, 4, 1), 32 >>>(X_d, W_map_delta2_div256_d,W_map_negative_delta2_div256_d, c_d)));
                                break;
                            }
                        }
                        if(M == 8 && K == 2560 && N == 2560){
                            if(S == 2048){
                                checkKernelErrors((rowWiseSplit2DeltaSmallAsync2<8, 2560, 2560, 2048, 32><<< dim3(2560, 8, 1), 32 >>>(X_d, W_map_delta2_div256_d,W_map_negative_delta2_div256_d, c_d)));
                                break;
                            }
                            if(S == 1536){
                                checkKernelErrors((rowWiseSplit2DeltaSmallAsync2<8, 2560, 2560, 1536, 32><<< dim3(2560, 8, 1), 32 >>>(X_d, W_map_delta2_div256_d,W_map_negative_delta2_div256_d, c_d)));
                                break;
                            }
                            if(S == 1024){
                                checkKernelErrors((rowWiseSplit2DeltaSmallAsync2<8, 2560, 2560, 1024, 32><<< dim3(2560, 8, 1), 32 >>>(X_d, W_map_delta2_div256_d,W_map_negative_delta2_div256_d, c_d)));
                                break;
                            }
                        }
                        if(M == 1 && K == 2560 && N == 3840){
                            if(S == 2048){
                                checkKernelErrors((rowWiseSplit2DeltaSmallAsync2<1, 2560, 3840, 2048, 32><<< dim3(3840, 1, 1), 32 >>>(X_d, W_map_delta2_div256_d,W_map_negative_delta2_div256_d, c_d)));
                                break;
                            }
                            if(S == 1536){
                                checkKernelErrors((rowWiseSplit2DeltaSmallAsync2<1, 2560, 3840, 1536, 32><<< dim3(3840, 1, 1), 32 >>>(X_d, W_map_delta2_div256_d,W_map_negative_delta2_div256_d, c_d)));
                                break;
                            }
                            if(S == 1024){
                                checkKernelErrors((rowWiseSplit2DeltaSmallAsync2<1, 2560, 3840, 1024, 32><<< dim3(3840, 1, 1), 32 >>>(X_d, W_map_delta2_div256_d,W_map_negative_delta2_div256_d, c_d)));
                                break;
                            }
                        }
                        if(M == 4 && K == 2560 && N == 3840){
                            if(S == 2048){
                                checkKernelErrors((rowWiseSplit2DeltaSmallAsync2<4, 2560, 3840, 2048, 32><<< dim3(3840, 4, 1), 32 >>>(X_d, W_map_delta2_div256_d,W_map_negative_delta2_div256_d, c_d)));
                                break;
                            }
                            if(S == 1536){
                                checkKernelErrors((rowWiseSplit2DeltaSmallAsync2<4, 2560, 3840, 1536, 32><<< dim3(3840, 4, 1), 32 >>>(X_d, W_map_delta2_div256_d,W_map_negative_delta2_div256_d, c_d)));
                                break;
                            }
                            if(S == 1024){
                                checkKernelErrors((rowWiseSplit2DeltaSmallAsync2<4, 2560, 3840, 1024, 32><<< dim3(3840, 4, 1), 32 >>>(X_d, W_map_delta2_div256_d,W_map_negative_delta2_div256_d, c_d)));
                                break;
                            }
                        }
                        if(M == 8 && K == 2560 && N == 3840){
                            if(S == 2048){
                                checkKernelErrors((rowWiseSplit2DeltaSmallAsync2<8, 2560, 3840, 2048, 32><<< dim3(3840, 8, 1), 32 >>>(X_d, W_map_delta2_div256_d,W_map_negative_delta2_div256_d, c_d)));
                                break;
                            }
                            if(S == 1536){
                                checkKernelErrors((rowWiseSplit2DeltaSmallAsync2<8, 2560, 3840, 1536, 32><<< dim3(3840, 8, 1), 32 >>>(X_d, W_map_delta2_div256_d,W_map_negative_delta2_div256_d, c_d)));
                                break;
                            }
                            if(S == 1024){
                                checkKernelErrors((rowWiseSplit2DeltaSmallAsync2<8, 2560, 3840, 1024, 32><<< dim3(3840, 8, 1), 32 >>>(X_d, W_map_delta2_div256_d,W_map_negative_delta2_div256_d, c_d)));
                                break;
                            }
                        }



                        // For 8B
                        if(M == 1 && K == 14336 && N == 4096) {
                            if(S == 8960){ // 40
                                checkKernelErrors((rowWiseSplit2DeltaSmallAsync2 < 1, 14336, 4096, 8960, 32, int><<< dim3(4096, 1, 1), dim3(32, 1, 1) >>>(X_d, W_map_delta2_div256_d,W_map_negative_delta2_div256_d, c_d)));
                                break;
                            }
                            if(S == 8704){ // 40
                                checkKernelErrors((rowWiseSplit2DeltaSmallAsync2 < 1, 14336, 4096, 8704, 32 ><<< dim3(4096, 1, 1), dim3(32, 1, 1) >>>(X_d, W_map_delta2_div256_d,W_map_negative_delta2_div256_d, c_d)));
                                break;
                            }
                            if(S == 5632){ // 60
                                checkKernelErrors((rowWiseSplit2DeltaSmallAsync2 < 1, 14336, 4096, 5632, 32 ><<< dim3(4096, 1, 1), dim3(32, 1, 1) >>>(X_d, W_map_delta2_div256_d,W_map_negative_delta2_div256_d, c_d)));
                                break;
                            }
                            if(S == 5376){ // 60
                                checkKernelErrors((rowWiseSplit2DeltaSmallAsync2 < 1, 14336, 4096, 5376, 32, int><<< dim3(4096, 1, 1), dim3(32, 1, 1) >>>(X_d, W_map_delta2_div256_d,W_map_negative_delta2_div256_d, c_d)));
                                break;
                            }
                        }
                        if(M == 1 && K == 4096 && N == 28672) {
                            if(S == 2560){ // 40
                                checkKernelErrors((rowWiseSplit2DeltaSmallAsync2 < 1, 4096, 28672, 2560, 32 ><<< dim3(28672, 1, 1), dim3(32, 1, 1) >>>(X_d, W_map_delta2_div256_d,W_map_negative_delta2_div256_d, c_d)));
                                break;
                            }
                            if(S == 1536){ // 60
                                checkKernelErrors((rowWiseSplit2DeltaSmallAsync2 < 1, 4096, 28672, 1536, 32 ><<< dim3(28672, 1, 1), dim3(32, 1, 1) >>>(X_d, W_map_delta2_div256_d,W_map_negative_delta2_div256_d, c_d)));
                                break;
                            }
                        }
                        if(M == 1 && K == 4096 && N == 4096) {
                            if(S == 2560){ // 40
                                checkKernelErrors((rowWiseSplit2DeltaSmallAsync2 < 1, 4096, 4096, 2560, 32 ><<< dim3(4096, 1, 1), dim3(32, 1, 1) >>>(X_d, W_map_delta2_div256_d,W_map_negative_delta2_div256_d, c_d)));
                                break;
                            }
                            if(S == 1536){ // 60
                                checkKernelErrors((rowWiseSplit2DeltaSmallAsync2 < 1, 4096, 4096, 1536, 32 ><<< dim3(4096, 1, 1), dim3(32, 1, 1) >>>(X_d, W_map_delta2_div256_d,W_map_negative_delta2_div256_d, c_d)));
                                break;
                            }
                        }
                        if(M == 1 && K == 4096 && N == 6144) {
                            if(S == 2560){ // 40
                                checkKernelErrors((rowWiseSplit2DeltaSmallAsync2 < 1, 4096, 6144, 2560, 32 ><<< dim3(6144, 1, 1), dim3(32, 1, 1) >>>(X_d, W_map_delta2_div256_d,W_map_negative_delta2_div256_d, c_d)));
                                break;
                            }
                            if(S == 1536){ // 60
                                checkKernelErrors((rowWiseSplit2DeltaSmallAsync2 < 1, 4096, 6144, 1536, 32 ><<< dim3(6144, 1, 1), dim3(32, 1, 1) >>>(X_d, W_map_delta2_div256_d,W_map_negative_delta2_div256_d, c_d)));
                                break;
                            }
                        }



                        if(M == 4 && K == 14336 && N == 4096) {
                            if(S == 8960){ // 40
                                checkKernelErrors((rowWiseSplit2DeltaSmallAsync2 < 4, 14336, 4096, 8960, 32, int><<< dim3(4096, 4, 1), dim3(32, 1, 1) >>>(X_d, W_map_delta2_div256_d,W_map_negative_delta2_div256_d, c_d)));
                                break;
                            }
                            if(S == 8704){ // 40
                                checkKernelErrors((rowWiseSplit2DeltaSmallAsync2 < 4, 14336, 4096, 8704, 32 ><<< dim3(4096, 4, 1), dim3(32, 1, 1) >>>(X_d, W_map_delta2_div256_d,W_map_negative_delta2_div256_d, c_d)));
                                break;
                            }
                            if(S == 5632){ // 60
                                checkKernelErrors((rowWiseSplit2DeltaSmallAsync2 < 4, 14336, 4096, 5632, 32 ><<< dim3(4096, 4, 1), dim3(32, 1, 1) >>>(X_d, W_map_delta2_div256_d,W_map_negative_delta2_div256_d, c_d)));
                                break;
                            }
                            if(S == 5376){ // 60
                                checkKernelErrors((rowWiseSplit2DeltaSmallAsync2 < 4, 14336, 4096, 5376, 32, int><<< dim3(4096, 4, 1), dim3(32, 1, 1) >>>(X_d, W_map_delta2_div256_d,W_map_negative_delta2_div256_d, c_d)));
                                break;
                            }
                        }
                        if(M == 4 && K == 4096 && N == 28672) {
                            if(S == 2560){ // 40
                                checkKernelErrors((rowWiseSplit2DeltaSmallAsync2 < 4, 4096, 28672, 2560, 32 ><<< dim3(28672, 4, 1), dim3(32, 1, 1) >>>(X_d, W_map_delta2_div256_d,W_map_negative_delta2_div256_d, c_d)));
                                break;
                            }
                            if(S == 1536){ // 60
                                checkKernelErrors((rowWiseSplit2DeltaSmallAsync2 < 4, 4096, 28672, 1536, 32 ><<< dim3(28672, 4, 1), dim3(32, 1, 1) >>>(X_d, W_map_delta2_div256_d,W_map_negative_delta2_div256_d, c_d)));
                                break;
                            }
                        }
                        if(M == 4 && K == 4096 && N == 4096) {
                            if(S == 2560){ // 40
                                checkKernelErrors((rowWiseSplit2DeltaSmallAsync2 < 4, 4096, 4096, 2560, 32 ><<< dim3(4096, 4, 1), dim3(32, 1, 1) >>>(X_d, W_map_delta2_div256_d,W_map_negative_delta2_div256_d, c_d)));
                                break;
                            }
                            if(S == 1536){ // 60
                                checkKernelErrors((rowWiseSplit2DeltaSmallAsync2 < 4, 4096, 4096, 1536, 32 ><<< dim3(4096, 4, 1), dim3(32, 1, 1) >>>(X_d, W_map_delta2_div256_d,W_map_negative_delta2_div256_d, c_d)));
                                break;
                            }
                        }
                        if(M == 4 && K == 4096 && N == 6144) {
                            if(S == 2560){ // 40
                                checkKernelErrors((rowWiseSplit2DeltaSmallAsync2 < 4, 4096, 6144, 2560, 32 ><<< dim3(6144, 4, 1), dim3(32, 1, 1) >>>(X_d, W_map_delta2_div256_d,W_map_negative_delta2_div256_d, c_d)));
                                break;
                            }
                            if(S == 1536){ // 60
                                checkKernelErrors((rowWiseSplit2DeltaSmallAsync2 < 4, 4096, 6144, 1536, 32 ><<< dim3(6144, 4, 1), dim3(32, 1, 1) >>>(X_d, W_map_delta2_div256_d,W_map_negative_delta2_div256_d, c_d)));
                                break;
                            }
                        }



                        if(M == 8 && K == 14336 && N == 4096) {
                            if(S == 8960){ // 40
                                checkKernelErrors((rowWiseSplit2DeltaSmallAsync2 < 8, 14336, 4096, 8960, 32, int><<< dim3(4096, 8, 1), dim3(32, 1, 1) >>>(X_d, W_map_delta2_div256_d,W_map_negative_delta2_div256_d, c_d)));
                                break;
                            }
                            if(S == 8704){ // 40
                                checkKernelErrors((rowWiseSplit2DeltaSmallAsync2 < 8, 14336, 4096, 8704, 32 ><<< dim3(4096, 8, 1), dim3(32, 1, 1) >>>(X_d, W_map_delta2_div256_d,W_map_negative_delta2_div256_d, c_d)));
                                break;
                            }
                            if(S == 5632){ // 60
                                checkKernelErrors((rowWiseSplit2DeltaSmallAsync2 < 8, 14336, 4096, 5632, 32 ><<< dim3(4096, 8, 1), dim3(32, 1, 1) >>>(X_d, W_map_delta2_div256_d,W_map_negative_delta2_div256_d, c_d)));
                                break;
                            }
                            if(S == 5376){ // 60
                                checkKernelErrors((rowWiseSplit2DeltaSmallAsync2 < 8, 14336, 4096, 5376, 32, int><<< dim3(4096, 8, 1), dim3(32, 1, 1) >>>(X_d, W_map_delta2_div256_d,W_map_negative_delta2_div256_d, c_d)));
                                break;
                            }
                        }
                        if(M == 8 && K == 4096 && N == 28672) {
                            if(S == 2560){ // 40
                                checkKernelErrors((rowWiseSplit2DeltaSmallAsync2 < 8, 4096, 28672, 2560, 32 ><<< dim3(28672, 8, 1), dim3(32, 1, 1) >>>(X_d, W_map_delta2_div256_d,W_map_negative_delta2_div256_d, c_d)));
                                break;
                            }
                            if(S == 1536){ // 60
                                checkKernelErrors((rowWiseSplit2DeltaSmallAsync2 < 8, 4096, 28672, 1536, 32 ><<< dim3(28672, 8, 1), dim3(32, 1, 1) >>>(X_d, W_map_delta2_div256_d,W_map_negative_delta2_div256_d, c_d)));
                                break;
                            }
                        }
                        if(M == 8 && K == 4096 && N == 4096) {
                            if(S == 2560){ // 40
                                checkKernelErrors((rowWiseSplit2DeltaSmallAsync2 < 8, 4096, 4096, 2560, 32 ><<< dim3(4096, 8, 1), dim3(32, 1, 1) >>>(X_d, W_map_delta2_div256_d,W_map_negative_delta2_div256_d, c_d)));
                                break;
                            }
                            if(S == 1536){ // 60
                                checkKernelErrors((rowWiseSplit2DeltaSmallAsync2 < 8, 4096, 4096, 1536, 32 ><<< dim3(4096, 8, 1), dim3(32, 1, 1) >>>(X_d, W_map_delta2_div256_d,W_map_negative_delta2_div256_d, c_d)));
                                break;
                            }
                        }
                        if(M == 8 && K == 4096 && N == 6144) {
                            if(S == 2560){ // 40
                                checkKernelErrors((rowWiseSplit2DeltaSmallAsync2 < 8, 4096, 6144, 2560, 32 ><<< dim3(6144, 8, 1), dim3(32, 1, 1) >>>(X_d, W_map_delta2_div256_d,W_map_negative_delta2_div256_d, c_d)));
                                break;
                            }
                            if(S == 1536){ // 60
                                checkKernelErrors((rowWiseSplit2DeltaSmallAsync2 < 8, 4096, 6144, 1536, 32 ><<< dim3(6144, 8, 1), dim3(32, 1, 1) >>>(X_d, W_map_delta2_div256_d,W_map_negative_delta2_div256_d, c_d)));
                                break;
                            }
                        }



                        STRICT_ASSERT(false, "Wrong split size");
                        break;
                    }
                    default: {
                        STRICT_ASSERT(false, "Wrong split size");
                    }
                }

                checkCudaErrors(cudaDeviceSynchronize());
            }
        });
        std::cout << ms / ((float) FLAGS_iter) << std::endl;
    }


    int8_t *W_i2s_d;

    if (FLAGS_i2s){
        STRICT_ASSERT(W_MAJOR == MAJOR_COL, "W must be col-major!\n");
        STRICT_ASSERT(X_MAJOR == MAJOR_ROW, "X must be row-major!\n");
        STRICT_ASSERT(C_MAJOR == MAJOR_ROW, "C must be row-major!\n");

        checkCudaErrors(cudaMalloc((void**) &W_i2s_d, K/ 4 * N));
        cudaMemset(W_i2s_d, 0, K / 4 * N);
        prepareW_i2s<<<N / 16, 16>>>(W_i2s_d, ctx_v);
        cudaDeviceSynchronize();

        cudaStream_t default_stream = 0;
        __nv_bfloat16 s = 0;

        ms = measureKernel([&]() {
            for (size_t i = 0; i < FLAGS_iter; i++) {
                if(M == 1 && N == 2560 && K == 2560){
                    checkKernelErrors((ladder_int8xint2_kernel<1, 2560, 2560, 8, 16><<< dim3(160, 1, 1), dim3(8,16,1), 0, default_stream >>>((int8_t *)X_d, W_i2s_d, c_d)));
                }else if (M == 2 && N == 2560 && K == 2560){
                    checkKernelErrors((ladder_int8xint2_kernel<2, 2560, 2560, 8, 16><<< dim3(160, 2, 1), dim3(8,16,1), 0, default_stream >>>((int8_t *)X_d, W_i2s_d, c_d)));
                }else if (M == 4 && N == 2560 && K == 2560){
                    checkKernelErrors((ladder_int8xint2_kernel<4, 2560, 2560, 8, 16><<< dim3(160, 4, 1), dim3(8,16,1), 0, default_stream >>>((int8_t *)X_d, W_i2s_d, c_d)));
                }else if (M == 4 && N == 2560 && K == 6912){
                    checkKernelErrors((ladder_int8xint2_kernel<4, 2560, 6912, 8, 16><<< dim3(160, 4, 1), dim3(8,16,1), 0, default_stream >>>((int8_t *)X_d, W_i2s_d, c_d)));
                }else if (M == 4 && N == 13824 && K == 2560){
                    checkKernelErrors((ladder_int8xint2_kernel<4, 13824, 2560, 8, 16><<< dim3(13824 / 16, 4, 1), dim3(8,16,1), 0, default_stream >>>((int8_t *)X_d, W_i2s_d, c_d)));
                }else if (M == 4 && N == 3840 && K == 2560){
                    checkKernelErrors((ladder_int8xint2_kernel<4, 3840, 2560, 8, 16><<< dim3(3840 / 16, 4, 1), dim3(8,16,1), 0, default_stream >>>((int8_t *)X_d, W_i2s_d, c_d)));


                }else if (M == 8 && N == 2560 && K == 6912){
                    checkKernelErrors((ladder_int8xint2_kernel<8, 2560, 6912, 8, 16><<< dim3(160, 8, 1), dim3(8,16,1), 0, default_stream >>>((int8_t *)X_d, W_i2s_d, c_d)));
                }else if (M == 8 && N == 2560 && K == 2560){
                    checkKernelErrors((ladder_int8xint2_kernel<8, 2560, 2560, 8, 16><<< dim3(160, 8, 1), dim3(8,16,1), 0, default_stream >>>((int8_t *)X_d, W_i2s_d, c_d)));
                }else if (M == 8 && N == 13824 && K == 2560){
                    checkKernelErrors((ladder_int8xint2_kernel<8, 13824, 2560, 8, 16><<< dim3(13824 / 16, 8, 1), dim3(8,16,1), 0, default_stream >>>((int8_t *)X_d, W_i2s_d, c_d)));
                }else if (M == 8 && N == 3840 && K == 2560){
                    checkKernelErrors((ladder_int8xint2_kernel<8, 3840, 2560, 8, 16><<< dim3(3840 / 16, 8, 1), dim3(8,16,1), 0, default_stream >>>((int8_t *)X_d, W_i2s_d, c_d)));




                }else if (M == 1 && N == 2560 && K == 6912){
                    checkKernelErrors((ladder_int8xint2_kernel<1, 2560, 6912, 8, 16><<< dim3(160, 1, 1), dim3(8,16,1), 0, default_stream >>>((int8_t *)X_d, W_i2s_d, c_d)));
                }else if (M == 1 && N == 3200 && K == 10240){
                    checkKernelErrors((ladder_int8xint2_kernel<1, 3200, 10240, 8, 16><<< dim3(200, 1, 1), dim3(8,16,1), 0, default_stream >>>((int8_t *)X_d, W_i2s_d, c_d)));
                }else if (M == 1 && N == 1280 && K == 10240){
                    checkKernelErrors((ladder_int8xint2_kernel<1, 1280, 10240, 8, 16><<< dim3(80, 1, 1), dim3(8,16,1), 0, default_stream >>>((int8_t *)X_d, W_i2s_d, c_d)));
                }else if (M == 1 && N == 640 && K == 10240){
                    checkKernelErrors((ladder_int8xint2_kernel<1, 640, 10240, 8, 16><<< dim3(40, 1, 1), dim3(8,16,1), 0, default_stream >>>((int8_t *)X_d, W_i2s_d, c_d)));
                }else if (M == 1 && N == 2560 && K == 10240){
                    checkKernelErrors((ladder_int8xint2_kernel<1, 2560, 10240, 8, 16><<< dim3(160, 1, 1), dim3(8,16,1), 0, default_stream >>>((int8_t *)X_d, W_i2s_d, c_d)));
                }else if (M == 1 && N == 13824 && K == 2560){
                    checkKernelErrors((ladder_int8xint2_kernel<1, 13824, 2560, 8, 16><<< dim3(864, 1, 1), dim3(8,16,1), 0, default_stream >>>((int8_t *)X_d, W_i2s_d, c_d)));
                }else if (M == 1 && N == 1024 && K == 4096){
                    checkKernelErrors((ladder_int8xint2_kernel<1, 1024, 4096, 8, 16><<< dim3(64, 1, 1), dim3(8,16,1), 0, default_stream >>>((int8_t *)X_d, W_i2s_d, c_d)));
                }else if (M == 1 && N == 1024 && K == 8192){
                    checkKernelErrors((ladder_int8xint2_kernel<1, 1024, 8192, 8, 16><<< dim3(64, 1, 1), dim3(8,16,1), 0, default_stream >>>((int8_t *)X_d, W_i2s_d, c_d)));
                }else if (M == 1 && N == 1024 && K == 10240){
                    checkKernelErrors((ladder_int8xint2_kernel<1, 1024, 8192, 8, 16><<< dim3(64, 1, 1), dim3(8,16,1), 0, default_stream >>>((int8_t *)X_d, W_i2s_d, c_d)));
                }else if (M == 1 && N == 768 && K == 8192){
                    checkKernelErrors((ladder_int8xint2_kernel<1, 768, 8192, 8, 16><<< dim3(48, 1, 1), dim3(8,16,1), 0, default_stream >>>((int8_t *)X_d, W_i2s_d, c_d)));
                }else if (M == 1 && N == 6912 && K == 2560){
                    checkKernelErrors((ladder_int8xint2_kernel<1, 6912, 2560, 8, 16><<< dim3(432, 1, 1), dim3(8,16,1), 0, default_stream >>>((int8_t *)X_d, W_i2s_d, c_d)));
                }else if (M == 1 && N == 640 && K == 2560){
                    checkKernelErrors((ladder_int8xint2_kernel<1, 640, 2560, 8, 16><<< dim3(40, 1, 1), dim3(8,16,1), 0, default_stream >>>((int8_t *)X_d, W_i2s_d, c_d)));
                }else if (M == 1 && N == 3840 && K == 2560){
                    checkKernelErrors((ladder_int8xint2_kernel<1, 3840, 2560, 8, 16><<< dim3(240, 1, 1), dim3(8,16,1), 0, default_stream >>>((int8_t *)X_d, W_i2s_d, c_d)));
                }else if (M == 1 && N == 512 && K == 1024){
                    checkKernelErrors((ladder_int8xint2_kernel<1, 512, 1024, 8, 16><<< dim3(512 / 16, 1, 1), dim3(8,16,1), 0, default_stream >>>((int8_t *)X_d, W_i2s_d, c_d)));
                }else if (M == 16 && N == 2560 && K == 6912){
                    checkKernelErrors((ladder_int8xint2_kernel<1, 2560, 6912, 8, 16><<< dim3(864, 16, 1), dim3(8,16,1), 0, default_stream >>>((int8_t *)X_d, W_i2s_d, c_d)));


                    // 8B case
                }else if (M == 1 && N == 4096 && K == 14336){
                    checkKernelErrors((ladder_int8xint2_kernel<1, 4096, 14336, 8, 16><<< dim3(4096 / 16, 1, 1), dim3(8,16,1), 0, default_stream >>>((int8_t *)X_d, W_i2s_d, c_d)));
                }else if (M == 1 && N == 28672 && K == 4096){
                    checkKernelErrors((ladder_int8xint2_kernel<1, 28672, 4096, 8, 16><<< dim3(28672 / 16, 1, 1), dim3(8,16,1), 0, default_stream >>>((int8_t *)X_d, W_i2s_d, c_d)));
                }else if (M == 1 && N == 4096 && K == 4096){
                    checkKernelErrors((ladder_int8xint2_kernel<1, 4096, 4096, 8, 16><<< dim3(4096 / 16, 1, 1), dim3(8,16,1), 0, default_stream >>>((int8_t *)X_d, W_i2s_d, c_d)));
                }else if (M == 1 && N == 6144 && K == 4096){
                    checkKernelErrors((ladder_int8xint2_kernel<1, 6144, 4096, 8, 16><<< dim3(6144 / 16, 1, 1), dim3(8,16,1), 0, default_stream >>>((int8_t *)X_d, W_i2s_d, c_d)));


                }else if (M == 4 && N == 4096 && K == 14336){
                    checkKernelErrors((ladder_int8xint2_kernel<4, 4096, 14336, 8, 16><<< dim3(4096 / 16, 4, 1), dim3(8,16,1), 0, default_stream >>>((int8_t *)X_d, W_i2s_d, c_d)));
                }else if (M == 4 && N == 28672 && K == 4096){
                    checkKernelErrors((ladder_int8xint2_kernel<4, 28672, 4096, 8, 16><<< dim3(28672 / 16, 4, 1), dim3(8,16,1), 0, default_stream >>>((int8_t *)X_d, W_i2s_d, c_d)));
                }else if (M == 4 && N == 4096 && K == 4096){
                    checkKernelErrors((ladder_int8xint2_kernel<4, 4096, 4096, 8, 16><<< dim3(4096 / 16, 4, 1), dim3(8,16,1), 0, default_stream >>>((int8_t *)X_d, W_i2s_d, c_d)));
                }else if (M == 4 && N == 6144 && K == 4096){
                    checkKernelErrors((ladder_int8xint2_kernel<4, 6144, 4096, 8, 16><<< dim3(6144 / 16, 4, 1), dim3(8,16,1), 0, default_stream >>>((int8_t *)X_d, W_i2s_d, c_d)));


                }else if (M == 8 && N == 4096 && K == 14336){
                    checkKernelErrors((ladder_int8xint2_kernel<8, 4096, 14336, 8, 16><<< dim3(4096 / 16, 8, 1), dim3(8,16,1), 0, default_stream >>>((int8_t *)X_d, W_i2s_d, c_d)));
                }else if (M == 8 && N == 28672 && K == 4096){
                    checkKernelErrors((ladder_int8xint2_kernel<8, 28672, 4096, 8, 16><<< dim3(28672 / 16, 8, 1), dim3(8,16,1), 0, default_stream >>>((int8_t *)X_d, W_i2s_d, c_d)));
                }else if (M == 8 && N == 4096 && K == 4096){
                    checkKernelErrors((ladder_int8xint2_kernel<8, 4096, 4096, 8, 16><<< dim3(4096 / 16, 8, 1), dim3(8,16,1), 0, default_stream >>>((int8_t *)X_d, W_i2s_d, c_d)));
                }else if (M == 8 && N == 6144 && K == 4096){
                    checkKernelErrors((ladder_int8xint2_kernel<8, 6144, 4096, 8, 16><<< dim3(6144 / 16, 8, 1), dim3(8,16,1), 0, default_stream >>>((int8_t *)X_d, W_i2s_d, c_d)));


                }else{
                    STRICT_ASSERT(false, "Wrong MKN size");
                }

                checkCudaErrors(cudaDeviceSynchronize());
            }
        });
        std::cout << ms / ((float) FLAGS_iter) << std::endl;
    }


    return 0;
}