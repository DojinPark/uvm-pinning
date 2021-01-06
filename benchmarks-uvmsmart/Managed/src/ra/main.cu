#include <iostream>
#include <math.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
//dojin
#include "../../common/util.h"
//

// #define N (1<<20)

__global__ void kernel(float* input, float* output, float* table, size_t size)
{
	int x_id = blockIdx.x * blockDim.x + threadIdx.x;
	if (x_id > size || x_id % 100 != 0)
    		return;

	float in_f = input[x_id];
	int in_i = (int)(floor(in_f));
	int table_index = (int)((in_f - float(in_i)) *( (float)(size) ));
	float* t = table + table_index;
	output[table_index] = t[0] * in_f;
}

int main(int argc, char ** argv)
{
  //int N = 1<<10;
  float *input, *output, *table;
  //dojin
  size_t N;
  bool pref = false;
  set_envs(&N, sizeof(float), argv[1], argv[2], &pref, argv[3]);
  N /= 3;
  //

  // Allocate Unified Memory -- accessible from CPU or GPU
  cudaMallocManaged(&input, N*sizeof(float));
  cudaMallocManaged(&output, N*sizeof(float));
  cudaMallocManaged(&table, N*sizeof(float));

  // initialize x and y arrays on the host
  for (int i = 0; i < N; i++) {
    input[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    //printf("-%lf-\n", input[i]);
    table[i] = ((float)(i));
  }

  int blockSize = 256;
  int numBlocks = (N + blockSize - 1) / blockSize;

  gpu_timer_set();
  kernel<<<numBlocks, blockSize>>>(input, output, table, N);

  cudaDeviceSynchronize();
  gpu_timer_pause();
  gpu_timer_record(argc, argv);

  // for (int i = 0; i < N; i++)
  //   if(output[i] != 0) {
  // 	printf("-%d %lf-\n", i, output[i]);
  //   }

  return 0;
}
