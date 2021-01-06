#!/usr/bin/make
include ../../conf.mk

default:
	nvcc -g -Wno-deprecated-gpu-targets $(UTIL_SRC) sssp.cu -lcudart -o sssp

prefetch:
	nvcc -g -DPREF -Wno-deprecated-gpu-targets $(UTIL_SRC) sssp.cu -lcudart -o sssp

clean:
	rm -rf sssp gpgpusim_power_report* _cuobjdump_* *.txt
