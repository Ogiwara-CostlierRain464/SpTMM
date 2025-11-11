/***************************************************************************
 * This file includes modifications based on the following work:
 * https://github.com/AlibabaResearch/flash-llm/blob/main/csrc/AsyncCopy_PTX.cuh
 *
 * Original code is licensed under the Apache License:
 *
 * Copyright 2023 The FLash-LLM Authors. All rights reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 ***************************************************************************/
// Extended from CUTLASS's source code

#include <cuda_fp16.h>
#include "macro.cuh"


template<int SizeInBytes>
__device__ __forceinline__ void cp_async(half* smem_ptr, const half* global_ptr, bool pred_guard = true)
{
    static_assert((SizeInBytes == 4 || SizeInBytes == 8 || SizeInBytes == 16), "Size is not supported");
    unsigned smem_int_ptr = __cvta_generic_to_shared(smem_ptr);
    asm volatile("{ \n"
                 "  .reg .pred p;\n"
                 "  setp.ne.b32 p, %0, 0;\n"
                 "  @p cp.async.cg.shared.global [%1], [%2], %3;\n"
                 "}\n" ::"r"((int)pred_guard),
    "r"(smem_int_ptr),
    "l"(global_ptr),
    "n"(SizeInBytes));
}

template<int SizeInBytes = 16>
__device__ __forceinline__ void cp_async(int4* smem_ptr, const int4* global_ptr, bool pred_guard = true)
{
    static_assert(SizeInBytes == 16, "Size is not supported");
    const unsigned smem_int_ptr = __cvta_generic_to_shared(smem_ptr);
    asm volatile("{ \n"
                 "  .reg .pred p;\n"
                 "  setp.ne.b32 p, %0, 0;\n"
                 "  @p cp.async.cg.shared.global [%1], [%2], %3;\n"
                 "}\n" ::"r"((int)pred_guard),
    "r"(smem_int_ptr),
    "l"(global_ptr),
    "n"(SizeInBytes));
}

template<int SizeInBytes = 8>
__device__ __forceinline__ void cp_async(int2* smem_ptr, const int2* global_ptr, bool pred_guard = true)
{
    static_assert((SizeInBytes == 4 || SizeInBytes == 8 || SizeInBytes == 16), "Size is not supported");
    const unsigned smem_int_ptr = __cvta_generic_to_shared(smem_ptr);
    asm volatile("{ \n"
                 "  .reg .pred p;\n"
                 "  setp.ne.b32 p, %0, 0;\n"
                 "  @p cp.async.ca.shared.global [%1], [%2], %3;\n"
                 "}\n" ::"r"((int)pred_guard),
    "r"(smem_int_ptr),
    "l"(global_ptr),
    "n"(SizeInBytes));
}

template<int SizeInBytes = 4>
__device__ __forceinline__ void cp_async(int* smem_ptr, const int* global_ptr, bool pred_guard = true)
{
    static_assert((SizeInBytes == 4), "Size is not supported");
    const unsigned smem_int_ptr = __cvta_generic_to_shared(smem_ptr);
    asm volatile("{ \n"
                 "  .reg .pred p;\n"
                 "  setp.ne.b32 p, %0, 0;\n"
                 "  @p cp.async.ca.shared.global [%1], [%2], %3;\n"
                 "}\n" ::"r"((int)pred_guard),
    "r"(smem_int_ptr),
    "l"(global_ptr),
    "n"(SizeInBytes));
}

// only used for kernel pipeline analysis
template<int SizeInBytes>
__device__ __forceinline__ void cp_async_test_only(half* smem_ptr, const half* global_ptr, bool pred_guard = true)
{
    static_assert((SizeInBytes == 4 || SizeInBytes == 8 || SizeInBytes == 16), "Size is not supported");
    unsigned smem_int_ptr = __cvta_generic_to_shared(smem_ptr);
    asm volatile("{ \n"
                 "  .reg .pred p;\n"
                 "  setp.ne.b32 p, %0, 0;\n"
                 "  @p cp.async.cg.shared.global [%1], [%2], %3, 0;\n"
                 "}\n" ::"r"((int)pred_guard),
    "r"(smem_int_ptr),
    "l"(global_ptr),
    "n"(SizeInBytes));
}

template<int SizeInBytes>
__device__ __forceinline__ void cp_async_ignore_src(half* smem_ptr, half* global_ptr)
{
    static_assert((SizeInBytes == 4 || SizeInBytes == 8 || SizeInBytes == 16), "Size is not supported");
    unsigned smem_int_ptr = __cvta_generic_to_shared(smem_ptr);
    asm volatile("{ \n"
                 "  cp.async.cg.shared.global [%0], [%1], %2, 0;\n"
                 "}\n" ::"r"(smem_int_ptr),
    "l"(global_ptr),
    "n"(SizeInBytes));
}

/// Establishes an ordering w.r.t previously issued cp.async instructions. Does not block.
__device__ __forceinline__ void cp_async_group_commit()
{
    asm volatile("cp.async.commit_group;\n" ::);
}

/// Blocks until all but <N> previous cp.async.commit_group operations have committed.
template<int N>
__device__ __forceinline__ void cp_async_wait_group()
{
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}


template<int I, int N>
__device__ __forceinline__ void cp_async_wait_group_unrolled() {
    if constexpr (I < N) {
        cp_async_wait_group<I>();
        cp_async_wait_group_unrolled<I + 1, N>();
    }
}

template<int N>
__device__ __forceinline__ void cp_async_wait_group_for() {
    cp_async_wait_group_unrolled<0, N>();
}

__device__ __forceinline__ void cp_async_wait_group_dynamic(int n) {
    switch (n) {
        case 0: cp_async_wait_group<0>(); break;
        case 1: cp_async_wait_group<1>(); break;
        case 2: cp_async_wait_group<2>(); break;
        case 3: cp_async_wait_group<3>(); break;
        case 4: cp_async_wait_group<4>(); break;
        case 5: cp_async_wait_group<5>(); break;
        case 6: cp_async_wait_group<6>(); break;
        case 7: cp_async_wait_group<7>(); break;
        case 8: cp_async_wait_group<8>(); break;
        case 9: cp_async_wait_group<9>(); break;
        case 10: cp_async_wait_group<10>(); break;
        case 11: cp_async_wait_group<11>(); break;
        case 12: cp_async_wait_group<12>(); break;
        case 13: cp_async_wait_group<13>(); break;
        case 14: cp_async_wait_group<14>(); break;
        case 15: cp_async_wait_group<15>(); break;
        case 16: cp_async_wait_group<16>(); break;
        case 17: cp_async_wait_group<17>(); break;
        case 18: cp_async_wait_group<18>(); break;
        case 19: cp_async_wait_group<19>(); break;
        case 20: cp_async_wait_group<20>(); break;
        case 21: cp_async_wait_group<21>(); break;
        case 22: cp_async_wait_group<22>(); break;
        case 23: cp_async_wait_group<23>(); break;
        case 24: cp_async_wait_group<24>(); break;
        case 25: cp_async_wait_group<25>(); break;
        case 26: cp_async_wait_group<26>(); break;
        case 27: cp_async_wait_group<27>(); break;
        case 28: cp_async_wait_group<28>(); break;
        case 29: cp_async_wait_group<29>(); break;
        case 30: cp_async_wait_group<30>(); break;
        case 31: cp_async_wait_group<31>(); break;
        case 32: cp_async_wait_group<32>(); break;

        default: asm("trap;");
    }
}

#define CP_ASYNC_WAIT_GROUP_PREDICATED_CASE(i) \
    asm volatile("{ .reg .pred p; setp.eq.s32 p, %0, " #i "; @p cp.async.wait_group " #i "; }\n" :: "r"(n));

// less BRANCH instruction but slower
__device__ __forceinline__ void cp_async_wait_group_predicated(int n) {
    CP_ASYNC_WAIT_GROUP_PREDICATED_CASE(0)
    CP_ASYNC_WAIT_GROUP_PREDICATED_CASE(1)
    CP_ASYNC_WAIT_GROUP_PREDICATED_CASE(2)
    CP_ASYNC_WAIT_GROUP_PREDICATED_CASE(3)
    CP_ASYNC_WAIT_GROUP_PREDICATED_CASE(4)
    CP_ASYNC_WAIT_GROUP_PREDICATED_CASE(5)
    CP_ASYNC_WAIT_GROUP_PREDICATED_CASE(6)
    CP_ASYNC_WAIT_GROUP_PREDICATED_CASE(7)
    CP_ASYNC_WAIT_GROUP_PREDICATED_CASE(8)
}

/// Blocks until all previous cp.async.commit_group operations have committed.
// cp.async.wait_all is equivalent to :
// cp.async.commit_group;
// cp.async.wait_group 0;
__device__ __forceinline__ void cp_async_wait_all()
{
    asm volatile("cp.async.wait_all;\n" ::);
}