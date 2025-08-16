#pragma once

#define MAJOR_ROW 0
#define MAJOR_COL 1
#define CAT(x, y) x ## y

#ifndef NDEBUG
__host__ __device__ void log_val(int val, const char* name) {
    printf("%s = %d ", name, val);
}
__host__  __device__ void log_val(char val, const char* name) {
    printf("%s = %d ", name, val);
}
__host__  __device__ void log_val(unsigned long val, const char* name) {
    printf("%s = %lu ", name, val);
}
__host__  __device__ void log_val(unsigned int val, const char* name) {
    printf("%s = %u ", name, val);
}
#endif


#ifndef NDEBUG
#define AT_0(mat, row_dim, col_dim, row, col)                                        \
    (*({                                                                             \
        uint64_t _r = (row);                                                             \
        uint64_t _c = (col);                                                             \
        uint64_t _rd = (row_dim);                                                        \
        uint64_t _cd = (col_dim);                                                        \
        if (_r >= _rd) {             \
            assert(false && "Row Out of Bound");                                         \
        }                                                                            \
        if (_c >= _cd) {             \
            assert(false && "Col Out of Bound");                                         \
        }                                                                            \
        &(mat[(uint64_t)_r * (uint64_t) _cd + (uint64_t) _c]);                                                       \
    }))
#else
    #define AT_0(mat, row_dim, col_dim, row, col) ((mat)[(uint64_t)(row) *  (uint64_t)(col_dim) + (uint64_t)(col)]) // NOTE: this could hurt performance
#endif

#ifndef NDEBUG
#define AT_1(mat, row_dim, col_dim, row, col)                                        \
    (*({                                                                             \
        auto _r = (row);                                                             \
        auto _c = (col);                                                             \
        auto _rd = (row_dim);                                                        \
        auto _cd = (col_dim);                                                        \
        if (_r >= _rd) {             \
        assert(false && "Row Out of Bound");                                         \
        }                                                                            \
        if (_c >= _cd) {             \
        assert(false && "Col Out of Bound");                                         \
        }                                                                           \
        &(mat[_c * _rd + _r]);                                                       \
    }))
#else
    #define AT_1(mat, row_dim, col_dim, row, col) ((mat)[(col) * (row_dim) + (row)])
#endif

#define AT(major) CAT(AT_, major)

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))
const int WARPSIZE = 32; // warpSize is not constexpr

#define STRICT_ASSERT(exp, msg) ({ do{ if(!(exp)){  fprintf(stderr, "%s\n", msg); abort();  } }while(0);})

// WARN: Do not use 0 as a seed!
__host__ __device__ uint32_t xorshift32(uint32_t *state) {
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}
