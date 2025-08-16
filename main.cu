#include <cuda.h>
#include <stdio.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <mma.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <cusparse.h>
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
#include <x86intrin.h>
//#define NDEBUG please insert from compiler option!
#include <cassert>


#include "macro.cuh"

DEFINE_uint64(row_split3, 0, "Run Row-wise SplitS 3 method with specified split size");
DEFINE_uint64(row_split_delta2, 0, "Run Row-wise Split Delta method 2 with specified split size");

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
      //.b = decide_W_map_width(K, S),
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
}


template <const uint M_, const uint K_, const uint N_, const uint S_, const uint SPLIT> // 32~256
__global__ void __launch_bounds__(SPLIT) rowWiseSplit3(
        const char* __restrict__ const X,
        const unsigned short* __restrict__ const W_map,
        const unsigned short* __restrict__ const W_map_negative,
        int* __restrict__ const C){

    static_assert(M_ == 1, "Only M==1 supported");
    static_assert((S_ / 2) % SPLIT == 0, "Should be completely devidable");

    const uint col = blockIdx.x;

    const ushort *W_map_base = &W_map[col * (S_ / 2)];
    const ushort *W_map_neg_base = &W_map_negative[col * (S_ / 2)];

    int accum = 0;

    for(uint row = threadIdx.x; row < (S_ / 2) ; row+=SPLIT){
        ushort pos_idx = __ldg(&W_map_base[row]);
        ushort neg_idx = __ldg(&W_map_neg_base[row]);

        accum += __ldg(&X[pos_idx]);
        accum -= __ldg(&X[neg_idx]);
    }

    __shared__ int sbuf[SPLIT];
    sbuf[threadIdx.x] = accum;
    __syncthreads();

    if constexpr (SPLIT >= 256){
        if(threadIdx.x < 128){
            sbuf[threadIdx.x] += sbuf[threadIdx.x + 128];
        }
        __syncthreads();
    }

    if constexpr (SPLIT >= 128){
        if(threadIdx.x < 64){
            sbuf[threadIdx.x] += sbuf[threadIdx.x + 64];
        }
        __syncthreads();
    }

    if(threadIdx.x < 32){
        if constexpr (SPLIT >= 64){
            accum = sbuf[threadIdx.x] + sbuf[threadIdx.x + 32];
        }
        if constexpr (SPLIT == 32){
            accum = sbuf[threadIdx.x];
        }

        accum = __reduce_add_sync(0xffffffff, accum);
    }

    if (threadIdx.x == 0) {
        C[col] = accum;
    }
}

template <const uint M_, const uint K_, const uint N_, const uint S_, const uint SPLIT> // 32~256
__global__ void __launch_bounds__(SPLIT) rowWiseSplit2Delta(
        const char* __restrict__ const X,
        const unsigned char* __restrict__ const W_map_delta,
        const unsigned char* __restrict__ const W_map_negative_delta,
        int* __restrict__ const C){

    static_assert(M_ == 1, "Only M==1 supported");
    static_assert((S_ / 2) % SPLIT == 0, "Should be completely devidable");

    const int land_id = threadIdx.x & 31; // threadIdx.x % 32
    const int warp_id = threadIdx.x >> 5; // threadIdx.x / 32
    const uint col = blockIdx.x;

    __shared__ int sbuf[SPLIT];
    ushort next_iter_pos = 0;
    ushort next_iter_neg = 0;

    int accum = 0;
    constexpr int NUM_WARPS = SPLIT / 32;

    const unsigned char *W_map_base = &W_map_delta[col * (S_ / 2)];
    const unsigned char *W_map_neg_base = &W_map_negative_delta[col * (S_ / 2)];

    for(u_int64_t row = threadIdx.x; row < (S_ / 2); row+=SPLIT){
        ushort delta_pos = __ldg(&W_map_base[row]);
        ushort delta_neg = __ldg(&W_map_neg_base[row]);

#pragma unroll
        for(int offset = 1; offset < 32; offset*=2){
            uint32_t packed = __shfl_up_sync(0xffffffff, ((uint32_t)delta_neg << 16) | (uint32_t)delta_pos, offset);

            delta_pos += (land_id >= offset) ? (ushort)(packed & 0xffff) : 0;
            delta_neg += (land_id >= offset) ? (ushort)((packed >> 16) & 0xffff) : 0;
        }

        // 2 shorts * 128
        // a thread access to the same ushort2
        ushort2 *lifter = reinterpret_cast<ushort2*>(sbuf);
        lifter[threadIdx.x] = make_ushort2(delta_pos, delta_neg);

        if(warp_id == 0){
            lifter[threadIdx.x].x += next_iter_pos;
            lifter[threadIdx.x].y += next_iter_neg;
#pragma unroll
            for(int i = 1; i < NUM_WARPS; i++){
                lifter[threadIdx.x + i * 32].x += lifter[i * 32 - 1].x;
                lifter[threadIdx.x + i * 32].y += lifter[i * 32 - 1].y;
            }
        }
        __syncthreads();

        next_iter_pos = lifter[NUM_WARPS * 32 - 1].x;
        next_iter_neg = lifter[NUM_WARPS * 32 - 1].y;

        accum += __ldg(&X[lifter[threadIdx.x].x]);
        accum -= __ldg(&X[lifter[threadIdx.x].y]);
        __syncthreads();
    }

    sbuf[threadIdx.x] = accum;
    __syncthreads();

    if constexpr (SPLIT >= 256){
        if(threadIdx.x < 128){
            sbuf[threadIdx.x] += sbuf[threadIdx.x + 128];
        }
        __syncthreads();
    }

    if constexpr (SPLIT >= 128){
        if(threadIdx.x < 64){
            sbuf[threadIdx.x] += sbuf[threadIdx.x + 64];
        }
        __syncthreads();
    }

    if(threadIdx.x < 32){
        if constexpr (SPLIT >= 64){
            accum = sbuf[threadIdx.x] + sbuf[threadIdx.x + 32];
        }
        if constexpr (SPLIT == 32){
            accum = sbuf[threadIdx.x];
        }

        accum = __reduce_add_sync(0xffffffff, accum);
    }

    if (threadIdx.x == 0) {
        C[col] = accum;
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
 * Using uniform distribution to make the range from -127 to 127
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


char *W_work_d;
unsigned short *W_map_d;
unsigned short *W_map_negative_d;
unsigned short *W_map_split_base_d;
unsigned short *W_map_split_negative_base_d;
unsigned char  *W_map_split_delta_d;
unsigned char  *W_map_split_negative_delta_d;
unsigned char  *W_map_delta2_d;
unsigned char  *W_map_negative_delta2_d;
unsigned char  *W_map_delta_warp_acc_d;
unsigned char  *W_map_negative_delta_warp_acc_d;

char  *W_vcsc_values_d;
unsigned short  *W_vcsc_counts_d;
unsigned short  *W_vcsc_indices_d;

if( FLAGS_row_split3 || FLAGS_row_split_delta2 ){
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

    checkCudaErrors(cudaMalloc((void**) &W_map_delta_warp_acc_d, sizeof(unsigned char) * (S / 2) * N));
    checkCudaErrors(cudaMalloc((void**) &W_map_negative_delta_warp_acc_d, sizeof(unsigned char) * (S / 2) * N));

    // only Column major
    checkCudaErrors(cudaMalloc((void**) &W_vcsc_values_d, sizeof(char) * 2 * N)); // only -1 & 1
    checkCudaErrors(cudaMalloc((void**) &W_vcsc_counts_d, sizeof(unsigned short) * 2 * N)); // only -1 & 1
    checkCudaErrors(cudaMalloc((void**) &W_vcsc_indices_d, sizeof(unsigned short) * S * N)); // only -1 & 1

    prepareW_map<<<N/16, 16>>>(W_work_d, W_map_d, W_map_negative_d, W_map_split_base_d, W_map_split_negative_base_d,
                               W_map_split_delta_d, W_map_split_negative_delta_d, W_map_delta2_d, W_map_negative_delta2_d, W_map_delta_warp_acc_d, W_map_negative_delta_warp_acc_d,
                               W_vcsc_values_d, W_vcsc_counts_d, W_vcsc_indices_d,
                                ctx_v);
    cudaDeviceSynchronize();
}


if(FLAGS_row_split3 > 0) {
    ms = measureKernel([&]() {
        for (size_t i = 0; i < FLAGS_iter; i++) {
            switch(FLAGS_row_split3){
                case 512: {
                    if(M == 1 && K == 6912 && N == 2560){
                        if(S == 5120){
                            checkKernelErrors((rowWiseSplit3<1, 6912, 2560, 5120, 512><<< (N * 512) * M / (L * 512), 512 >>>(X_d, W_map_d,W_map_negative_d, c_d)));
                            break;
                        }
                        if(S == 4096){
                            checkKernelErrors((rowWiseSplit3<1, 6912, 2560, 4096, 512><<< (N * 512) * M / (L * 512), 512 >>>(X_d, W_map_d,W_map_negative_d, c_d)));
                            break;
                        }
                        if(S == 3072){
                            checkKernelErrors((rowWiseSplit3<1, 6912, 2560, 3072, 512><<< (N * 512) * M / (L * 512), 512 >>>(X_d, W_map_d,W_map_negative_d, c_d)));
                            break;
                        }
                        if(S == 2048){
                            checkKernelErrors((rowWiseSplit3<1, 6912, 2560, 2048, 512><<< (N * 512) * M / (L * 512), 512 >>>(X_d, W_map_d,W_map_negative_d, c_d)));
                            break;
                        }
                        if(S == 1024){
                            checkKernelErrors((rowWiseSplit3<1, 6912, 2560, 1024, 512><<< (N * 512) * M / (L * 512), 512 >>>(X_d, W_map_d,W_map_negative_d, c_d)));
                            break;
                        }
                    }
                    if(M == 1 && K == 2560 && N == 2560){
                        if(S == 2048){
                            checkKernelErrors((rowWiseSplit3<1, 2560, 2560, 2048, 512><<< (N * 512) * M / (L * 512), 512 >>>(X_d, W_map_d,W_map_negative_d, c_d)));
                            break;
                        }
                        if(S == 1024){
                            checkKernelErrors((rowWiseSplit3<1, 2560, 2560, 1024, 512><<< (N * 512) * M / (L * 512), 512 >>>(X_d, W_map_d,W_map_negative_d, c_d)));
                            break;
                        }
                    }
                    if(M == 1 && K == 2560 && N == 13824){
                        if(S == 2048){
                            checkKernelErrors((rowWiseSplit3<1, 2560, 13824, 2048, 512><<< (N * 512) * M / (L * 512), 512 >>>(X_d, W_map_d,W_map_negative_d, c_d)));
                            break;
                        }
                        if(S == 1024){
                            checkKernelErrors((rowWiseSplit3<1, 2560, 13824, 1024, 512><<< (N * 512) * M / (L * 512), 512 >>>(X_d, W_map_d,W_map_negative_d, c_d)));
                            break;
                        }
                    }
                    if(M == 1 && K == 2560 && N == 3840){
                        if(S == 2048){
                            checkKernelErrors((rowWiseSplit3<1, 2560, 3840, 2048, 512><<< (N * 512) * M / (L * 512), 512 >>>(X_d, W_map_d,W_map_negative_d, c_d)));
                            break;
                        }
                        if(S == 1024){
                            checkKernelErrors((rowWiseSplit3<1, 2560, 3840, 1024, 512><<< (N * 512) * M / (L * 512), 512 >>>(X_d, W_map_d,W_map_negative_d, c_d)));
                            break;
                        }
                    }
                    STRICT_ASSERT(false, "Wrong split size");
                }
                case 256: {
                    if(M == 1 && K == 6912 && N == 2560){
                        if(S == 5120){
                            checkKernelErrors((rowWiseSplit3<1, 6912, 2560, 5120, 256><<< (N * 256) * M / (L * 256), 256 >>>(X_d, W_map_d,W_map_negative_d, c_d)));
                            break;
                        }
                        if(S == 4608){
                            checkKernelErrors((rowWiseSplit3<1, 6912, 2560, 4608, 256><<< (N * 256) * M / (L * 256), 256 >>>(X_d, W_map_d,W_map_negative_d, c_d)));
                            break;
                        }
                        if(S == 4096){
                            checkKernelErrors((rowWiseSplit3<1, 6912, 2560, 4096, 256><<< (N * 256) * M / (L * 256), 256 >>>(X_d, W_map_d,W_map_negative_d, c_d)));
                            break;
                        }
                        if(S == 3072){
                             checkKernelErrors((rowWiseSplit3<1, 6912, 2560, 3072, 256><<< (N * 256) * M / (L * 256), 256 >>>(X_d, W_map_d,W_map_negative_d, c_d)));
                             break;
                        }
                        if(S == 2048){
                            checkKernelErrors((rowWiseSplit3<1, 6912, 2560, 2048, 256><<< (N * 256) * M / (L * 256), 256 >>>(X_d, W_map_d,W_map_negative_d, c_d)));
                            break;
                        }
                        if(S == 1024){
                            checkKernelErrors((rowWiseSplit3<1, 6912, 2560, 1024, 256><<< (N * 256) * M / (L * 256), 256 >>>(X_d, W_map_d,W_map_negative_d, c_d)));
                            break;
                        }
                        if(S == 512){
                            checkKernelErrors((rowWiseSplit3<1, 6912, 2560, 512, 256><<< (N * 256) * M / (L * 256), 256 >>>(X_d, W_map_d,W_map_negative_d, c_d)));
                            break;
                        }
                    }
                    if(M == 1 && K == 2560 && N == 2560){
                        if(S == 2048){
                            checkKernelErrors((rowWiseSplit3<1, 2560, 2560, 2048, 256><<< (N * 256) * M / (L * 256), 256 >>>(X_d, W_map_d,W_map_negative_d, c_d)));
                            break;
                        }
                        if(S == 1536){
                            checkKernelErrors((rowWiseSplit3<1, 2560, 2560, 1536, 256><<< (N * 256) * M / (L * 256), 256 >>>(X_d, W_map_d,W_map_negative_d, c_d)));
                            break;
                        }
                        if(S == 1024){
                            checkKernelErrors((rowWiseSplit3<1, 2560, 2560, 1024, 256><<< (N * 256) * M / (L * 256), 256 >>>(X_d, W_map_d,W_map_negative_d, c_d)));
                            break;
                        }
                        if(S == 512){
                            checkKernelErrors((rowWiseSplit3<1, 2560, 2560, 512, 256><<< (N * 256) * M / (L * 256), 256 >>>(X_d, W_map_d,W_map_negative_d, c_d)));
                            break;
                        }
                    }
                    if(M == 1 && K == 2560 && N == 13824){
                        if(S == 2048){
                            checkKernelErrors((rowWiseSplit3<1, 2560, 13824, 2048, 256><<< (N * 256) * M / (L * 256), 256 >>>(X_d, W_map_d,W_map_negative_d, c_d)));
                            break;
                        }
                        if(S == 1536){
                            checkKernelErrors((rowWiseSplit3<1, 2560, 13824, 1536, 256><<< (N * 256) * M / (L * 256), 256 >>>(X_d, W_map_d,W_map_negative_d, c_d)));
                            break;
                        }
                        if(S == 1024){
                            checkKernelErrors((rowWiseSplit3<1, 2560, 13824, 1024, 256><<< (N * 256) * M / (L * 256), 256 >>>(X_d, W_map_d,W_map_negative_d, c_d)));
                            break;
                        }
                        if(S == 512){
                            checkKernelErrors((rowWiseSplit3<1, 2560, 13824, 512, 256><<< (N * 256) * M / (L * 256), 256 >>>(X_d, W_map_d,W_map_negative_d, c_d)));
                            break;
                        }
                    }
                    if(M == 1 && K == 2560 && N == 3840){
                        if(S == 2048){
                            checkKernelErrors((rowWiseSplit3<1, 2560, 3840, 2048, 256><<< (N * 256) * M / (L * 256), 256 >>>(X_d, W_map_d,W_map_negative_d, c_d)));
                            break;
                        }
                        if(S == 1536){
                            checkKernelErrors((rowWiseSplit3<1, 2560, 3840, 1536, 256><<< (N * 256) * M / (L * 256), 256 >>>(X_d, W_map_d,W_map_negative_d, c_d)));
                            break;
                        }
                        if(S == 1024){
                            checkKernelErrors((rowWiseSplit3<1, 2560, 3840, 1024, 256><<< (N * 256) * M / (L * 256), 256 >>>(X_d, W_map_d,W_map_negative_d, c_d)));
                            break;
                        }
                        if(S == 512){
                            checkKernelErrors((rowWiseSplit3<1, 2560, 3840, 512, 256><<< (N * 256) * M / (L * 256), 256 >>>(X_d, W_map_d,W_map_negative_d, c_d)));
                            break;
                        }
                    }
                    STRICT_ASSERT(false, "Wrong split size");
                }
                case 192: {
                    if(M == 1 && K == 6912 && N == 2560){
                        if(S == 1536){
                            checkKernelErrors((rowWiseSplit3<1, 6912, 2560, 1536, 192><<< 2560, 192 >>>(X_d, W_map_d,W_map_negative_d, c_d)));
                            break;
                        }
                    }
                    if(M == 1 && K == 2560 && N == 2560){

                        if(S == 1536){
                            checkKernelErrors((rowWiseSplit3<1, 2560, 2560, 1536, 192><<< 2560, 192 >>>(X_d, W_map_d,W_map_negative_d, c_d)));
                            break;
                        }

                    }
                    if(M == 1 && K == 2560 && N == 13824){

                        if(S == 1536){
                            checkKernelErrors((rowWiseSplit3<1, 2560, 13824, 1536, 192><<< 13824, 192 >>>(X_d, W_map_d,W_map_negative_d, c_d)));
                            break;
                        }

                    }
                    if(M == 1 && K == 2560 && N == 3840){

                        if(S == 1536){
                            checkKernelErrors((rowWiseSplit3<1, 2560, 3840, 1536, 192><<< 3840, 192 >>>(X_d, W_map_d,W_map_negative_d, c_d)));
                            break;
                        }

                    }
                    STRICT_ASSERT(false, "Wrong split size");
                }
                case 128: {
                    if(M == 1 && K == 6912 && N == 2560){
                        if(S == 5120){
                            checkKernelErrors((rowWiseSplit3<1, 6912, 2560, 5120, 128><<< (N * 128) * M / (L * 128), 128>>>(X_d, W_map_d,W_map_negative_d, c_d)));
                            break;
                        }
                        if(S == 4608){
                            checkKernelErrors((rowWiseSplit3<1, 6912, 2560, 4608, 128><<< (N * 128) * M / (L * 128), 128 >>>(X_d, W_map_d,W_map_negative_d, c_d)));
                            break;
                        }
                        if(S == 4096){
                            checkKernelErrors((rowWiseSplit3<1, 6912, 2560, 4096, 128><<< (N * 128) * M / (L * 128), 128 >>>(X_d, W_map_d,W_map_negative_d, c_d)));
                            break;
                        }
                        if(S == 3072){
                            checkKernelErrors((rowWiseSplit3<1, 6912, 2560, 3072, 128><<< (N * 128) * M / (L * 128), 128 >>>(X_d, W_map_d,W_map_negative_d, c_d)));
                            break;
                        }
                        if(S == 2048){
                            checkKernelErrors((rowWiseSplit3<1, 6912, 2560, 2048, 128><<< (N * 128) * M / (L * 128), 128 >>>(X_d, W_map_d,W_map_negative_d, c_d)));
                            break;
                        }
                        if(S == 1024){
                            checkKernelErrors((rowWiseSplit3<1, 6912, 2560, 1024, 128><<< (N * 128) * M / (L * 128), 128 >>>(X_d, W_map_d,W_map_negative_d, c_d)));
                            break;
                        }
                        if(S == 512){
                            checkKernelErrors((rowWiseSplit3<1, 6912, 2560, 512, 128><<< (N * 128) * M / (L * 128), 128 >>>(X_d, W_map_d,W_map_negative_d, c_d)));
                            break;
                        }
                    }
                    if(M == 1 && K == 2560 && N == 2560){
                        if(S == 2048){
                            checkKernelErrors((rowWiseSplit3<1, 2560, 2560, 2048, 128><<< (N * 128) * M / (L * 128), 128 >>>(X_d, W_map_d,W_map_negative_d, c_d)));
                            break;
                        }
                        if(S == 1536){
                            checkKernelErrors((rowWiseSplit3<1, 2560, 2560, 1536, 128><<< (N * 128) * M / (L * 128), 128 >>>(X_d, W_map_d,W_map_negative_d, c_d)));
                            break;
                        }
                        if(S == 1024){
                            checkKernelErrors((rowWiseSplit3<1, 2560, 2560, 1024, 128><<< (N * 128) * M / (L * 128), 128 >>>(X_d, W_map_d,W_map_negative_d, c_d)));
                            break;
                        }
                        if(S == 512){
                            checkKernelErrors((rowWiseSplit3<1, 2560, 2560, 512, 128><<< (N * 128) * M / (L * 128), 128 >>>(X_d, W_map_d,W_map_negative_d, c_d)));
                            break;
                        }
                    }
                    if(M == 1 && K == 2560 && N == 13824){
                        if(S == 2048){
                            checkKernelErrors((rowWiseSplit3<1, 2560, 13824, 2048, 128><<< 13824, 128 >>>(X_d, W_map_d,W_map_negative_d, c_d)));
                            break;
                        }
                        if(S == 1536){
                            checkKernelErrors((rowWiseSplit3<1, 2560, 13824, 1536, 128><<< (N * 128) * M / (L * 128), 128 >>>(X_d, W_map_d,W_map_negative_d, c_d)));
                            break;
                        }
                        if(S == 1024){
                            checkKernelErrors((rowWiseSplit3<1, 2560, 13824, 1024, 128><<< (N * 128) * M / (L * 128), 128 >>>(X_d, W_map_d,W_map_negative_d, c_d)));
                            break;
                        }
                        if(S == 512){
                            checkKernelErrors((rowWiseSplit3<1, 2560, 13824, 512, 128><<< (N * 128) * M / (L * 128), 128 >>>(X_d, W_map_d,W_map_negative_d, c_d)));
                            break;
                        }
                    }
                    if(M == 1 && K == 2560 && N == 3840){
                        if(S == 2048){
                            checkKernelErrors((rowWiseSplit3<1, 2560, 3840, 2048, 128><<< 3840, 128 >>>(X_d, W_map_d,W_map_negative_d, c_d)));
                            break;
                        }
                        if(S == 1536){
                            checkKernelErrors((rowWiseSplit3<1, 2560, 3840, 1536, 128><<< (N * 128) * M / (L * 128), 128 >>>(X_d, W_map_d,W_map_negative_d, c_d)));
                            break;
                        }
                        if(S == 1024){
                            checkKernelErrors((rowWiseSplit3<1, 2560, 3840, 1024, 128><<< (N * 128) * M / (L * 128), 128 >>>(X_d, W_map_d,W_map_negative_d, c_d)));
                            break;
                        }
                        if(S == 512){
                            checkKernelErrors((rowWiseSplit3<1, 2560, 3840, 512, 128><<< (N * 128) * M / (L * 128), 128 >>>(X_d, W_map_d,W_map_negative_d, c_d)));
                            break;
                        }
                    }
                    STRICT_ASSERT(false, "Wrong split size");
                }
                case 96: {
                    if(M == 1 && K == 6912 && N == 2560){
                        if(S == 3072){
                            checkKernelErrors((rowWiseSplit3<1, 6912, 2560, 3072, 96><<< (N * 96) * M / (L * 96), 96 >>>(X_d, W_map_d,W_map_negative_d, c_d)));
                            break;
                        }
                        if(S == 1536){
                            checkKernelErrors((rowWiseSplit3<1, 6912, 2560, 1536, 96><<< (N * 96) * M / (L * 96), 96 >>>(X_d, W_map_d,W_map_negative_d, c_d)));
                            break;
                        }
                    }
                    if(M == 1 && K == 2560 && N == 2560){
                        if(S == 1536){
                            checkKernelErrors((rowWiseSplit3<1, 2560, 2560, 1536, 96><<< (N * 96) * M / (L * 96), 96 >>>(X_d, W_map_d,W_map_negative_d, c_d)));
                            break;
                        }
                    }
                    if(M == 1 && K == 2560 && N == 13824){
                        if(S == 1536){
                            checkKernelErrors((rowWiseSplit3<1, 2560, 13824, 1536, 96><<< (N * 96) * M / (L * 96), 96 >>>(X_d, W_map_d,W_map_negative_d, c_d)));
                            break;
                        }
                    }
                    if(M == 1 && K == 2560 && N == 3840){
                        if(S == 1536){
                            checkKernelErrors((rowWiseSplit3<1, 2560, 3840, 1536, 96><<< (N * 96) * M / (L * 96), 96 >>>(X_d, W_map_d,W_map_negative_d, c_d)));
                            break;
                        }
                    }
                    STRICT_ASSERT(false, "Wrong split size");
                }
                case 64: {
                    if(M == 1 && K == 6912 && N == 2560){
                        if(S == 5120){
                            checkKernelErrors((rowWiseSplit3<1, 6912, 2560, 5120, 64><<< (N * 64) * M / (L * 64), 64 >>>(X_d, W_map_d,W_map_negative_d, c_d)));
                            break;
                        }
                        if(S == 4096){
                            checkKernelErrors((rowWiseSplit3<1, 6912, 2560, 4096, 64><<< (N * 64) * M / (L * 64), 64 >>>(X_d, W_map_d,W_map_negative_d, c_d)));
                            break;
                        }
                        if(S == 3072){
                            checkKernelErrors((rowWiseSplit3<1, 6912, 2560, 3072, 64><<< (N * 64) * M / (L * 64), 64 >>>(X_d, W_map_d,W_map_negative_d, c_d)));
                            break;
                        }
                        if(S == 2048){
                            checkKernelErrors((rowWiseSplit3<1, 6912, 2560, 2048, 64><<< (N * 64) * M / (L * 64), 64 >>>(X_d, W_map_d,W_map_negative_d, c_d)));
                            break;
                        }
                        if(S == 1024){
                            checkKernelErrors((rowWiseSplit3<1, 6912, 2560, 1024, 64><<< (N * 64) * M / (L * 64), 64 >>>(X_d, W_map_d,W_map_negative_d, c_d)));
                            break;
                        }
                        if(S == 512){
                            checkKernelErrors((rowWiseSplit3<1, 6912, 2560, 512, 64><<< (N * 64) * M / (L * 64), 64 >>>(X_d, W_map_d,W_map_negative_d, c_d)));
                            break;
                        }
                    }
                    if(M == 1 && K == 2560 && N == 2560){
                        if(S == 2048){
                            checkKernelErrors((rowWiseSplit3<1, 2560, 2560, 2048, 64><<< (N * 64) * M / (L * 64), 64 >>>(X_d, W_map_d,W_map_negative_d, c_d)));
                            break;
                        }
                        if(S == 1536){
                            checkKernelErrors((rowWiseSplit3<1, 2560, 2560, 1536, 64><<< (N * 64) * M / (L * 64), 64 >>>(X_d, W_map_d,W_map_negative_d, c_d)));
                            break;
                        }
                        if(S == 1024){
                            checkKernelErrors((rowWiseSplit3<1, 2560, 2560, 1024, 64><<< (N * 64) * M / (L * 64), 64 >>>(X_d, W_map_d,W_map_negative_d, c_d)));
                            break;
                        }
                        if(S == 512){
                            checkKernelErrors((rowWiseSplit3<1, 2560, 2560, 512, 64><<< (N * 64) * M / (L * 64), 64 >>>(X_d, W_map_d,W_map_negative_d, c_d)));
                            break;
                        }
                    }
                    if(M == 1 && K == 2560 && N == 13824){
                        if(S == 2048){
                            checkKernelErrors((rowWiseSplit3<1, 2560, 13824, 2048, 64><<< (N * 64) * M / (L * 64), 64 >>>(X_d, W_map_d,W_map_negative_d, c_d)));
                            break;
                        }
                        if(S == 1536){
                            checkKernelErrors((rowWiseSplit3<1, 2560, 13824, 1536, 64><<< (N * 64) * M / (L * 64), 64 >>>(X_d, W_map_d,W_map_negative_d, c_d)));
                            break;
                        }
                        if(S == 1024){
                            checkKernelErrors((rowWiseSplit3<1, 2560, 13824, 1024, 64><<< (N * 64) * M / (L * 64), 64 >>>(X_d, W_map_d,W_map_negative_d, c_d)));
                            break;
                        }
                        if(S == 512){
                            checkKernelErrors((rowWiseSplit3<1, 2560, 13824, 512, 64><<< (N * 64) * M / (L * 64), 64 >>>(X_d, W_map_d,W_map_negative_d, c_d)));
                            break;
                        }
                    }
                    if(M == 1 && K == 2560 && N == 3840){
                        if(S == 2048){
                            checkKernelErrors((rowWiseSplit3<1, 2560, 3840, 2048, 64><<< (N * 64) * M / (L * 64), 64 >>>(X_d, W_map_d,W_map_negative_d, c_d)));
                            break;
                        }
                        if(S == 1536){
                            checkKernelErrors((rowWiseSplit3<1, 2560, 3840, 1536, 64><<< (N * 64) * M / (L * 64), 64 >>>(X_d, W_map_d,W_map_negative_d, c_d)));
                            break;
                        }
                        if(S == 1024){
                            checkKernelErrors((rowWiseSplit3<1, 2560, 3840, 1024, 64><<< (N * 64) * M / (L * 64), 64 >>>(X_d, W_map_d,W_map_negative_d, c_d)));
                            break;
                        }
                        if(S == 512){
                            checkKernelErrors((rowWiseSplit3<1, 2560, 3840, 512, 64><<< (N * 64) * M / (L * 64), 64 >>>(X_d, W_map_d,W_map_negative_d, c_d)));
                            break;
                        }
                    }
                    STRICT_ASSERT(false, "Wrong split size");
                }
                case 32: {
                    if(M == 1 && K == 6912 && N == 2560){
                        if(S == 5120){
                            checkKernelErrors((rowWiseSplit3<1, 6912, 2560, 5120, 32><<< (N * 32) * M / (L * 32), 32 >>>(X_d, W_map_d,W_map_negative_d, c_d)));
                            break;
                        }
                        if(S == 4096){
                            checkKernelErrors((rowWiseSplit3<1, 6912, 2560, 4096, 32><<< (N * 32) * M / (L * 32), 32 >>>(X_d, W_map_d,W_map_negative_d, c_d)));
                            break;
                        }
                        if(S == 3072){
                             checkKernelErrors((rowWiseSplit3<1, 6912, 2560, 3072, 32><<< (N * 32) * M / (L * 32), 32 >>>(X_d, W_map_d,W_map_negative_d, c_d)));
                             break;
                        }
                        if(S == 2048){
                            checkKernelErrors((rowWiseSplit3<1, 6912, 2560, 2048, 32><<< (N * 32) * M / (L * 32), 32 >>>(X_d, W_map_d,W_map_negative_d, c_d)));
                            break;
                        }
                        if(S == 1024){
                            checkKernelErrors((rowWiseSplit3<1, 6912, 2560, 1024, 32><<< (N * 32) * M / (L * 32), 32 >>>(X_d, W_map_d,W_map_negative_d, c_d)));
                            break;
                        }
                        if(S == 512){
                            checkKernelErrors((rowWiseSplit3<1, 6912, 2560, 512, 32><<< (N * 32) * M / (L * 32), 32 >>>(X_d, W_map_d,W_map_negative_d, c_d)));
                            break;
                        }
                    }
                    if(M == 1 && K == 2560 && N == 2560){
                        if(S == 2048){
                            checkKernelErrors((rowWiseSplit3<1, 2560, 2560, 2048, 32><<< (N * 32) * M / (L * 32), 32 >>>(X_d, W_map_d,W_map_negative_d, c_d)));
                            break;
                        }
                        if(S == 1024){
                            checkKernelErrors((rowWiseSplit3<1, 2560, 2560, 1024, 32><<< (N * 32) * M / (L * 32), 32 >>>(X_d, W_map_d,W_map_negative_d, c_d)));
                            break;
                        }
                        if(S == 512){
                            checkKernelErrors((rowWiseSplit3<1, 2560, 2560, 512, 32><<< (N * 32) * M / (L * 32), 32 >>>(X_d, W_map_d,W_map_negative_d, c_d)));
                            break;
                        }
                    }
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

if(FLAGS_row_split_delta2 > 0) {
    ms = measureKernel([&]() {
        for (size_t i = 0; i < FLAGS_iter; i++) {
            switch(FLAGS_row_split_delta2){
                case 128: {
                    if(M == 1 && K == 6912 && N == 2560){
                        if(S == 5120){
                            checkKernelErrors((rowWiseSplit2Delta<1, 6912, 2560, 5120, 128><<< 2560, 128 >>>(X_d, W_map_delta2_d,W_map_negative_delta2_d, c_d)));
                            break;
                        }
                        if(S == 4096){
                            checkKernelErrors((rowWiseSplit2Delta<1, 6912, 2560, 4096, 128><<< 2560, 128 >>>(X_d, W_map_delta2_d,W_map_negative_delta2_d, c_d)));
                            break;
                        }
                        if(S == 3072){
                            checkKernelErrors((rowWiseSplit2Delta<1, 6912, 2560, 3072, 128><<< 2560, 128 >>>(X_d, W_map_delta2_d,W_map_negative_delta2_d, c_d)));
                            break;
                        }
                        if(S == 2048){
                            checkKernelErrors((rowWiseSplit2Delta<1, 6912, 2560, 2048, 128><<< 2560, 128 >>>(X_d, W_map_delta2_d,W_map_negative_delta2_d, c_d)));
                            break;
                        }
                    }
                    STRICT_ASSERT(false, "Wrong split size");
                    break;
                }
                case 64: {
                    if(M == 1 && K == 6912 && N == 2560){
                        if(S == 4096){
                            checkKernelErrors((rowWiseSplit2Delta<1, 6912, 2560, 4096, 64><<< 2560, 64 >>>(X_d, W_map_delta2_d,W_map_negative_delta2_d, c_d)));
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

    return 0;
}