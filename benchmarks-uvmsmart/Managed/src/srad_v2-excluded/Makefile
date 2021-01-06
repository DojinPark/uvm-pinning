#!/usr/bin/make
include ../../conf.mk

# CUDA compiler
NVCC = nvcc
NVCC_FLAGS = -g -Wno-deprecated-gpu-targets -w

default:
	$(NVCC) $(NVCC_FLAGS) $(KERNEL_DIM) $(UTIL_SRC) srad.cu -lcudart -o srad

prefetch:
	$(NVCC) $(NVCC_FLAGS) $(KERNEL_DIM) $(UTIL_SRC) -DPREF srad.cu -lcudart -o srad

clean: $(SRC)
	rm -f srad *.txt gpgpusim_power_report* _cuobjdump_*
