#!/usr/bin/make
include ../../conf.mk

# CUDA compiler
NVCC = nvcc
NVCC_FLAGS = -g -Wno-deprecated-gpu-targets -w

default:
	$(NVCC) $(NVCC_FLAGS) $(UTIL_SRC) pathfinder.cu -lcudart -o pathfinder

prefetch:
	$(NVCC) $(NVCC_FLAGS) $(UTIL_SRC) -DPREF pathfinder.cu -lcudart -o pathfinder

clean:
	rm -f pathfinder *.txt gpgpusim_power_report* _cuobjdump_*
