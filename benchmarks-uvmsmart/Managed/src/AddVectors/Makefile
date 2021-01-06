#!/usr/bin/make
include ../../conf.mk

default:
	nvcc -g -Wno-deprecated-gpu-targets add_vectors.cu $(UTIL_SRC) -lcudart -o addvectors

# prefetch:
# 	nvcc -g -DPREF -Wno-deprecated-gpu-targets add_vectors.cu -lcudart -o addvectors

clean:
	rm -rf add_vectors gpgpusim_power_report* _cuobjdump_* *.txt
