// Courtesy of https://devblogs.nvidia.com/parallelforall/unified-memory-cuda-beginners/
// REMOVE ME: Uncommnet the code only upon full implementation or get seg-fault
 
#include <iostream>
#include <math.h>
#include "../../common/util.h"
 
// CUDA kernel to add elements of two arrays
__global__
void add(int n, float *x, float *y)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride)
    y[i] = x[i] + y[i];
}
 
int main(int argc, char ** argv)
{
occupy_gpu_space(argv[1]);
//dojin
bool pref = false;

  int N = 1<<20;
  float *x, *y;
 
  // Allocate Unified Memory -- accessible from CPU or GPU
  cudaMallocManaged(&x, N*sizeof(float));
  cudaMallocManaged(&y, N*sizeof(float));
 
  // initialize x and y arrays on the host
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  cudaStream_t stream1;
  cudaStream_t stream2;
  cudaStream_t stream3;
if (pref) {
  // Prefetch the data to the GPU
  int device = -1;
  cudaGetDevice(&device);

  cudaStreamCreate(&stream1);
  cudaStreamCreate(&stream2);
  cudaStreamCreate(&stream3);

  cudaMemPrefetchAsync(x, N*sizeof(float), device, stream1);
  cudaMemPrefetchAsync(y, N*sizeof(float), device, stream2);
}
  // Launch kernel on 1M elements on the GPU
  int blockSize = 256;
  int numBlocks = (N + blockSize - 1) / blockSize;
  
gpu_timer_set();
if (pref)
  add<<<numBlocks, blockSize, 0, stream3>>>(N, x, y);
else
  add<<<numBlocks, blockSize>>>(N, x, y);
  
  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();
gpu_timer_pause();
gpu_timer_record(argc, argv);
 
  // Check for errors (all values should be 3.0f)
  //float maxError = 0.0f;
  //for (int i = 0; i < N; i++)
  //  maxError = fmax(maxError, fabs(y[i]-3.0f));
  //std::cout << "Max error: " << maxError << std::endl;
 
  // Free memory
  cudaFree(x);
  cudaFree(y);
 
  return 0;
}

