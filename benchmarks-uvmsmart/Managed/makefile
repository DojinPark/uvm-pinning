include conf.mk
SHELL:=/bin/bash
DIRS := AddVectors FDTD-2D nw pathfinder ra 
BINS := addvectors fdtd nw pathfinder ra 
#Excluded
# hotspot backprop bfs stencil srad_v2 sssp 
# hotspot backprop bfs stencil srad sssp 

all:
	dirs=($(DIRS)); \
	bins=($(BINS));	\
	for i in $${!dirs[@]}; do \
		echo ----- building $${dirs[$$i]} -------; \
		make -C ./src/$${dirs[$$i]} default; \
		cp ./src/$${dirs[$$i]}/$${bins[$$i]} ./bin/; \
		echo;  \
	done

clean:
	for dir in $(DIRS); do \
		make -C ./src/$$dir clean; \
	done; \
	for bin in $(BINS); do \
		rm -f ./bin/$$bin; \
	done;
