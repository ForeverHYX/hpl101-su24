TARGET = gemm
SOURCE = baseline.cu

NVCC = nvcc
NVCCFLAGS += -lcublas -O3 -cudart=shared -Xcompiler -fopenmp -arch=sm_75

$(TARGET):$(SOURCE)
	$(NVCC) $(SOURCE) -o $(TARGET) $(NVCCFLAGS)

.PHONY:clean
clean:
	rm -rf $(TARGET)
