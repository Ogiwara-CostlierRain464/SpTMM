#!/bin/bash

out_name=${1:-rcr}
out="../build/${out_name}"

DEFINES="-DNDEBUG -DX_MAJOR=MAJOR_ROW -DW_MAJOR=MAJOR_COL -DC_MAJOR=MAJOR_ROW"

nvcc --std=c++20 \
    -gencode=arch=compute_90,code=\"sm_90,compute_90\" \
    -x cu -Xptxas -O3 ../main.cu \
    -o "$out" -I/root/cutlass/include --expt-relaxed-constexpr -lgflags -lnvidia-ml -lcusparse -lcublas $DEFINES