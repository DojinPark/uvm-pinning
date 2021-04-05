#!/usr/bin/make
include ../../conf.mk

# CUDA compiler
NVCC = nvcc
NVCC_FLAGS = -g -Wno-deprecated-gpu-targets -w

SRC = hotspot.cu

EXE = hotspot

default:
	$(NVCC) $(NVCC_FLAGS) $(KERNEL_DIM) $(UTIL_SRC) hotspot.cu -lcudart -o hotspot

prefetch:
	$(NVCC) $(NVCC_FLAGS) $(KERNEL_DIM) -DPREF $(UTIL_SRC) hotspot.cu -lcudart -o hotspot

clean:
	rm -f hotspot *.txt gpgpusim_power_report* _cuobjdump_*
