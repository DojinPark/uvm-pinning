#!/usr/bin/make
include ../../conf.mk

default:
	nvcc -g -Wno-deprecated-gpu-targets -w fdtd2d.cu -lcudart $(UTIL_SRC) -I../../common/eigen3/include/eigen3 -o fdtd

clean:
	rm -rf fdtd gpgpusim_power_report* _cuobjdump_* *.txt
