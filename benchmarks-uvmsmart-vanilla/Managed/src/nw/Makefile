#!/usr/bin/make
include ../../conf.mk

# CUDA compiler
NVCC = nvcc
NVCC_FLAGS = -g -Wno-deprecated-gpu-targets -w

default:
	$(NVCC) $(NVCC_FLAGS) ${KERNEL_DIM} $(UTIL_SRC) needle.cu -lcudart -o nw

prefetch:
	$(NVCC) $(NVCC_FLAGS) ${KERNEL_DIM} $(UTIL_SRC) -DPREF needle.cu -lcudart -o nw

clean:
	rm -f nw *.txt gpgpusim_power_report* _cuobjdump_*
