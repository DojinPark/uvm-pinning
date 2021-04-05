#!/usr/bin/make
include ../../conf.mk
# CUDA compiler
NVCC = nvcc
NVCC_FLAGS = -g -Wno-deprecated-gpu-targets -w

default:
	$(NVCC) $(NVCC_FLAGS) $(NVCC_FLAGS) $(UTIL_SRC) bfs.cu -lcudart -o bfs 

prefetch:
	$(NVCC) $(NVCC_FLAGS) $(NVCC_FLAGS) $(UTIL_SRC) -DPREF bfs.cu -lcudart -o bfs 

clean:
	rm -f bfs *.linkinfo *.txt gpgpusim_power_report* _cuobjdump_*
