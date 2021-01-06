#!/usr/bin/make
include ../../conf.mk

# C compiler
CC = gcc
CC_FLAGS = -g  -O2

# CUDA compiler
NVCC = nvcc
NVCC_FLAGS = -g -Wno-deprecated-gpu-targets

default:
	$(CC) $(CC_FLAGS) file.cc -c
	$(NVCC) $(NVCC_FLAGS) -c kernels.cu
	$(NVCC) $(NVCC_FLAGS) -c main.cu
	$(NVCC) $(NVCC_FLAGS) $(UTIL_SRC) main.o kernels.o file.o -lcudart -lm -o stencil

prefetch:
	$(CC) $(CC_FLAGS) file.cc -c
	$(NVCC) $(NVCC_FLAGS) -DPREF -c kernels.cu
	$(NVCC) $(NVCC_FLAGS) -DPREF -c main.cu
	$(NVCC) $(NVCC_FLAGS) $(UTIL_SRC) -DPREF main.o kernels.o file.o -lcudart -lm -o stencil
clean:
	rm -f *.o *~ stencil *.linkinfo *.txt gpgpusim_power_report* _cuobjdump_*
