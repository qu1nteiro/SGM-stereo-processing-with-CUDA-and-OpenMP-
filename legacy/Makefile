
.PHONY: all clean

TARGETS = sgmCuda sgmOpenMP testDiffs

all: $(TARGETS)

sgmCuda: sgmCuda.cu
	nvcc -arch=sm_30 -O3 -ICommon $^ -o $@

sgmOpenMP: sgmOpenMP.c
	g++ -fopenmp -O3 -ICommon $^ -o $@

testDiffs: testDiffs.c
	g++ -O3 -ICommon $^ -o $@

clean:
	rm -f $(TARGETS) 
