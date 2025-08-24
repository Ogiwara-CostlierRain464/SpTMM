/**
* This file includes modifications based on the following work:
 * https://github.com/microsoft/BitNet/blob/main/gpu/bitnet_kernels/bitnet_kernels.h
 *
 * Original code is licensed under the MIT License:
*     MIT License

    Copyright (c) Microsoft Corporation.

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE
 *
 */

#include <cuda_runtime.h>
#include <math_constants.h>
#include <math.h>
#include <mma.h>
#include <iostream>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>


#if (((__CUDACC_VER_MAJOR__ == 11) && (__CUDACC_VER_MINOR__ >= 4)) || (__CUDACC_VER_MAJOR__ > 11))
#define TVM_ENABLE_L2_PREFETCH 1
#else
#define TVM_ENABLE_L2_PREFETCH 0
#endif

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 800
#define TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST 1
#else
#define TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST 0
#endif

__device__ void decode_i2s_to_i8s(int *_i2s, signed char *_i8s, const int N = 16) // int[1] 32bit == 16elems -> char[16] == 16 elems
{
    // convert 8 int2b_t to 8 int8b_t -> 2 int32
    uint *i8s = reinterpret_cast<uint *>(_i8s);

    // i2s = {e0, e4, e8, e12, e1, e5, e9, e13, e2, e6, e10, e14, e3, e7, e11, e15}
    uint const i2s = *_i2s;

    static constexpr uint immLut = (0xf0 & 0xcc) | 0xaa;     // 0b11101010  // (a and b) or c
    static constexpr uint BOTTOM_MASK = 0x03030303;          // 0xf -> 0b11 select 0,3 //  0000 0011 | 0000 0011 | 0000 0011 | 0000 0011
    static constexpr uint I4s_TO_I8s_MAGIC_NUM = 0x00000000; // not used

#pragma unroll
    for (int i = 0; i < (N / 4); i++)
    {
        asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
                : "=r"(i8s[i])
                : "r"(i2s >> (2 * i)), "n"(BOTTOM_MASK), "n"(I4s_TO_I8s_MAGIC_NUM), "n"(immLut));
        i8s[i] = __vsubss4(i8s[i], 0x02020202);
    }
}


template <int M, int N, int K, int K_block_size, int N_block_size>
__global__ void __launch_bounds__(128) ladder_int8xint2_kernel(int8_t* __restrict__ A, int8_t* __restrict__ B, int* __restrict__ dtype_transform) {
    constexpr int K_per_loop = 16;
    constexpr int wmma_K = 32;
    constexpr int wmma_N = 16;
    int in_thread_C_local[1];
    alignas(16) signed char A_local[K_per_loop];
    int B_reshape_local[1];
    signed char B_decode_local[K_per_loop];
    int red_buf0[1];
    in_thread_C_local[0] = 0;
    #pragma unroll
    for (int k_0 = 0; k_0 < K/(K_per_loop * K_block_size); ++k_0) { // split-K
        *(int4*)(A_local + 0) = *(int4*)(A + ((k_0 * K_per_loop * K_block_size) + (((int)threadIdx.x) * K_per_loop)));
        B_reshape_local[0] = *(int*)(B +
            (((int)blockIdx.x) * N_block_size * K / 4) +
            (k_0 * K_block_size * K_per_loop * wmma_N / 4) +
            ((((int)threadIdx.x) >> 1) * wmma_K * wmma_N / 4) +
            ((((int)threadIdx.y) >> 3) * (wmma_K * wmma_N / 2) / 4) +
            ((((int)threadIdx.x) & 1) * (wmma_K * wmma_N / 4) / 4) +
            ((((int)threadIdx.y) & 7) * (wmma_K / 2) / 4)
            );
        decode_i2s_to_i8s(B_reshape_local, B_decode_local, 16);
        #pragma unroll
        for (int k_2_0 = 0; k_2_0 < 4; ++k_2_0) {
            in_thread_C_local[0] = __dp4a(*(int *)&A_local[((k_2_0 * 4))],*(int *)&B_decode_local[((k_2_0 * 4))], in_thread_C_local[0]);
        }
    }
    red_buf0[0] = in_thread_C_local[0];
    #pragma unroll
    for (int offset = K_block_size/2; offset > 0; offset /= 2) {
        red_buf0[0] += __shfl_down_sync(__activemask(), red_buf0[0], offset, K_block_size);
    }
    int out_idx = ((((int)blockIdx.x) * N_block_size) + ((int)threadIdx.y));
    if (threadIdx.x == 0)
        dtype_transform[out_idx] = red_buf0[0];
}