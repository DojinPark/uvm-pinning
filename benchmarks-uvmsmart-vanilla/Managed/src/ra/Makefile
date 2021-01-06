#!/usr/bin/make
include ../../conf.mk

default:
	nvcc -g -Wno-deprecated-gpu-targets $(UTIL_SRC) main.cu -lcudart -o ra

clean:
	rm -rf ra gpgpusim_power_report* _cuobjdump_* *.txt
