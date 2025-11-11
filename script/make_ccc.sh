#!/bin/bash

out_name=${1:-ccc}
out="../build/${out_name}"

DEFINES="-DNDEBUG -DX_MAJOR=MAJOR_COL -DW_MAJOR=MAJOR_COL -DC_MAJOR=MAJOR_COL"

nvcc --std=c++20 \
    -gencode=arch=compute_90,code=\"sm_90,compute_90\" \
    -x cu -Xptxas -O3 ../main.cu -I/root/cutlass/include \
    -o "$out" --expt-relaxed-constexpr -lgflags -lnvidia-ml -lcusparse -lcublas $DEFINES