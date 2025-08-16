#!/bin/bash

DEFINES="-DNDEBUG -DX_MAJOR=MAJOR_COL -DW_MAJOR=MAJOR_COL -DC_MAJOR=MAJOR_COL"

# コンパイルコマンド
nvcc --std=c++17 \
    -gencode=arch=compute_80,code=\"sm_80,compute_80\" \
    -x cu -Xptxas -O3 ../main.cu \
    -o ../build/ccc -lgflags $DEFINES