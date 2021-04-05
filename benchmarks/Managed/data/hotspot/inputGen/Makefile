FLAGS := -Wno-write-strings

all: hotspotex hotspotver

hotspotex: hotspotex.cpp
	g++ $(FLAGS) -L../../../common -lutil -o $@ $<

hotspotver: hotspotver.cpp
	g++ $(FLAGS) -o $@ $<

clean:
	rm hotspotex hotspotver

