#!/usr/bin/make
include ../../conf.mk
# CUDA compiler
NVCC = nvcc
NVCC_FLAGS = -g -Wno-deprecated-gpu-targets -w

default:
	$(NVCC) $(NVCC_FLAGS) $(KERNEL_DIM) backprop.cu $(UTIL_SRC) -lcudart -o backprop

# prefetch:
# 	$(NVCC) $(NVCC_FLAGS) $(KERNEL_DIM) -DPREF backprop.cu $(UTIL_SRC) -lcudart -o backprop

clean: $(SRC)
	rm -f backprop *.txt gpgpusim_power_report* _cuobjdump_*
