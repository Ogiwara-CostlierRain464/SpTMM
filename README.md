# About
Implementation of `SpTMM: Multiplying Ternary Matrix Without Multiplication for Transformers`

# Reproduction instruction


```bash
# Step1: clone and start container with H100
git clone https://github.com/Ogiwara-CostlierRain464/SpTMM
docker build -t tmm_image .
docker run --cap-add SYS_ADMIN --privileged -it -v ./:/work -d --gpus all --name tmm tmm_image bash

# Step2: Inside of the container, build CUDA Kernel
docker exec -it tmm /bin/bash
cd work
mkdir build && cd build
source ../script/make_ccc.sh

# Step3: Run SpTMM, SpTMM-delta, cuBLAS, and bitnet.cpp
./ccc -row_split3 128 -M=1 -K=6912 -N=2560 -S=4096 -iter=100000 # SpTMM with Split-K=128
./ccc -row_split_delta2 128 -M=1 -K=6912 -N=2560 -S=4096 -iter=100000 # SpTMM-delta with Split-K=128
./ccc -cu_blas -M=1 -K=6912 -N=2560 -S=4096 -iter=100000 # cuBLAS with Split-K=128
./ccc -i2s -M=1 -K=6912 -N=2560 -S=4096 -iter=100000 # bitnet.cpp with Split-K=128
```


