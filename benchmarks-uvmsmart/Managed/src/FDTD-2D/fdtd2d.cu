/**
 * fdtd2d.cu: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <unistd.h>
#include <sys/time.h>
#include <cuda.h>
//dojin
#include "../../common/util.h"
#include <unsupported/Eigen/Polynomials>
#include <iostream>
//

//#include "../../common/polybenchUtilFuncts.h"

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 10.05

#define GPU_DEVICE 0

/* Problem size */
//#define tmax 500
//#define nx 2048
//#define ny 2048

// #define tmax 5
// #define nx 1200
// #define ny 1200


/* Thread block dimensions */
#define DIM_THREAD_BLOCK_X 32
#define DIM_THREAD_BLOCK_Y 32

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;



void init_arrays(DATA_TYPE* _fict_, DATA_TYPE* ex, DATA_TYPE* ey, DATA_TYPE* hz, size_t nx, size_t ny, unsigned tmax)
{
	int i, j;

  	for (i = 0; i < tmax; i++)
	{
		_fict_[i] = (DATA_TYPE) i;
	}
	
	for (i = 0; i < nx; i++)
	{
		for (j = 0; j < ny; j++)
		{
			ex[i*ny + j] = ((DATA_TYPE) i*(j+1) + 1) / nx;
			ey[i*ny + j] = ((DATA_TYPE) (i-1)*(j+2) + 2) / nx;
			hz[i*ny + j] = ((DATA_TYPE) (i-9)*(j+4) + 3) / nx;
		}
	}
}


void runFdtd(DATA_TYPE* _fict_, DATA_TYPE* ex, DATA_TYPE* ey, DATA_TYPE* hz, size_t nx, size_t ny, unsigned tmax)
{
	int t, i, j;
	
	for (t=0; t < tmax; t++)  
	{
		for (j=0; j < ny; j++)
		{
			ey[0*ny + j] = _fict_[t];
		}
	
		for (i = 1; i < nx; i++)
		{
       		for (j = 0; j < ny; j++)
			{
       			ey[i*ny + j] = ey[i*ny + j] - 0.5*(hz[i*ny + j] - hz[(i-1)*ny + j]);
        		}
		}

		for (i = 0; i < nx; i++)
		{
       		for (j = 1; j < ny; j++)
			{
				ex[i*(ny+1) + j] = ex[i*(ny+1) + j] - 0.5*(hz[i*ny + j] - hz[i*ny + (j-1)]);
			}
		}

		for (i = 0; i < nx; i++)
		{
			for (j = 0; j < ny; j++)
			{
				hz[i*ny + j] = hz[i*ny + j] - 0.7*(ex[i*(ny+1) + (j+1)] - ex[i*(ny+1) + j] + ey[(i+1)*ny + j] - ey[i*ny + j]);
			}
		}
	}
}

/*
void compareResults(DATA_TYPE* hz1, DATA_TYPE* hz2)
{
	int i, j, fail;
	fail = 0;
	
	for (i=0; i < nx; i++) 
	{
		for (j=0; j < ny; j++) 
		{
			if (percentDiff(hz1[i*ny + j], hz2[i*ny + j]) > PERCENT_DIFF_ERROR_THRESHOLD) 
			{
				fail++;
			}
		}
	}
	
	// Print results
	printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
}


void GPU_argv_init()
{
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, GPU_DEVICE);
	printf("setting device %d with name %s\n",GPU_DEVICE,deviceProp.name);
	cudaSetDevice( GPU_DEVICE );
}
*/


__global__ void fdtd_step1_kernel(DATA_TYPE* _fict_, DATA_TYPE *ex, DATA_TYPE *ey, DATA_TYPE *hz, int t, size_t nx, size_t ny)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	// size_t j = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
	// size_t i = (size_t)blockIdx.y * (size_t)blockDim.y + (size_t)threadIdx.y;

	if ((i < nx) && (j < ny))
	{
		if (i == 0) 
		{
			ey[i * ny + j] = _fict_[t];
		}
		else
		{ 
			ey[i * ny + j] = ey[i * ny + j] - 0.5f*(hz[i * ny + j] - hz[(i-1) * ny + j]);
		}
	}
}


__global__ void fdtd_step2_kernel(DATA_TYPE *ex, DATA_TYPE *ey, DATA_TYPE *hz, int t, size_t nx, size_t ny)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	// size_t j = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
	// size_t i = (size_t)blockIdx.y * (size_t)blockDim.y + (size_t)threadIdx.y;
	
	if ((i < nx) && (j < ny) && (j > 0))
	{
		ex[i * (ny+1) + j] = ex[i * (ny+1) + j] - 0.5f*(hz[i * ny + j] - hz[i * ny + (j-1)]);
	}
}


__global__ void fdtd_step3_kernel(DATA_TYPE *ex, DATA_TYPE *ey, DATA_TYPE *hz, int t, size_t nx, size_t ny)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	// size_t j = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
	// size_t i = (size_t)blockIdx.y * (size_t)blockDim.y + (size_t)threadIdx.y;
	
	if ((i < nx) && (j < ny))
	{	
		hz[i * ny + j] = hz[i * ny + j] - 0.7f*(ex[i * (ny+1) + (j+1)] - ex[i * (ny+1) + j] + ey[(i + 1) * ny + j] - ey[i * ny + j]);
	}
}


void fdtdCuda(DATA_TYPE* _fict_, DATA_TYPE* ex, DATA_TYPE* ey, DATA_TYPE* hz, size_t nx, size_t ny, unsigned tmax)//, DATA_TYPE* hz_outputFromGpu)
{
	//double t_start, t_end;
/*
	DATA_TYPE *_fict_gpu;
	DATA_TYPE *ex_gpu;
	DATA_TYPE *ey_gpu;
	DATA_TYPE *hz_gpu;

	cudaMalloc((void **)&_fict_gpu, sizeof(DATA_TYPE) * tmax);
	cudaMalloc((void **)&ex_gpu, sizeof(DATA_TYPE) * nx * (ny + 1));
	cudaMalloc((void **)&ey_gpu, sizeof(DATA_TYPE) * (nx + 1) * ny);
	cudaMalloc((void **)&hz_gpu, sizeof(DATA_TYPE) * nx * ny);

	cudaMemcpy(_fict_gpu, _fict_, sizeof(DATA_TYPE) * tmax, cudaMemcpyHostToDevice);
	cudaMemcpy(ex_gpu, ex, sizeof(DATA_TYPE) * nx * (ny + 1), cudaMemcpyHostToDevice);
	cudaMemcpy(ey_gpu, ey, sizeof(DATA_TYPE) * (nx + 1) * ny, cudaMemcpyHostToDevice);
	cudaMemcpy(hz_gpu, hz, sizeof(DATA_TYPE) * nx * ny, cudaMemcpyHostToDevice);
*/

	//dojin-note
	// block(32, 32)  is the max by compute capability
	// ny / 32 = max 65535
	// than max(ny) is 65535*32 = 2097120
	dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
	dim3 grid( (size_t)ceil(((float)ny) / ((float)block.x)), (size_t)ceil(((float)nx) / ((float)block.y)));

	//t_start = rtclock();

	for(int t = 0; t< tmax; t++)
	{
		fdtd_step1_kernel<<<grid,block>>>(_fict_, ex, ey, hz, t, nx, ny);
		cudaThreadSynchronize();
		fdtd_step2_kernel<<<grid,block>>>(ex, ey, hz, t, nx, ny);
		cudaThreadSynchronize();
		fdtd_step3_kernel<<<grid,block>>>(ex, ey, hz, t, nx, ny);
		cudaThreadSynchronize();
	}
	
	//t_end = rtclock();
    	//fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);

	//cudaMemcpy(hz_outputFromGpu, hz_gpu, sizeof(DATA_TYPE) * nx * ny, cudaMemcpyDeviceToHost);	
		
	//cudaFree(_fict_gpu);
	//cudaFree(ex_gpu);
	//cudaFree(ey_gpu);
	//cudaFree(hz_gpu);
}

int main(int argc, char ** argv)
{
	//double t_start, t_end;

	DATA_TYPE* _fict_;
	DATA_TYPE* ex;
	DATA_TYPE* ey;
	DATA_TYPE* hz;
	//DATA_TYPE* hz_outputFromGpu;

/*
	_fict_ = (DATA_TYPE*)malloc(tmax*sizeof(DATA_TYPE));
	ex = (DATA_TYPE*)malloc(nx*(ny+1)*sizeof(DATA_TYPE));
	ey = (DATA_TYPE*)malloc((nx+1)*ny*sizeof(DATA_TYPE));
	hz = (DATA_TYPE*)malloc(nx*ny*sizeof(DATA_TYPE));
*/	

	//dojin
	size_t n;
	unsigned tmax = 5;  // The number of iterations

	set_envs(&n, sizeof(float), argv[1], argv[2], 0, 0);

	Eigen::PolynomialSolver<double, Eigen::Dynamic> solver;
	Eigen::VectorXd coeff(3);
	coeff[0] = tmax - (double)n; coeff[1] = 2; coeff[2] = 3;
	solver.compute(coeff);
	const Eigen::PolynomialSolver<double, Eigen::Dynamic>::RootsType & r = solver.roots();
	
	n = static_cast<unsigned int>(round(r[0].real()));
	size_t nx = n, ny = n;
	printf("nx = ny = %u\n", n);

	size_t dim_test = static_cast<size_t>(((float)ny) / ((float)DIM_THREAD_BLOCK_X));
	if (dim_test > CUDA_MAX_BLOCK_DIM) {
		printf("Reached maximum dimension %llu > %d\n", n, dim_test);
		printf("Try dividing the data size by the factor %f\n", (float)n/dim_test);
		return 1;
	}

	//
	
	printf("-------------Size: %lf MB--------------\n", 
		(float)(sizeof(DATA_TYPE)*(tmax + nx*(ny+1)+(nx+1)*ny+nx*ny))/1024.0/1024.0);

	cudaMallocManaged(&_fict_, tmax*sizeof(DATA_TYPE));
	cudaMallocManaged(&ex, nx*(ny+1)*sizeof(DATA_TYPE));
	cudaMallocManaged(&ey, (nx+1)*ny*sizeof(DATA_TYPE));
	cudaMallocManaged(&hz, nx*ny*sizeof(DATA_TYPE));
	
	//hz_outputFromGpu = (DATA_TYPE*)malloc(nx*ny*sizeof(DATA_TYPE));

	init_arrays(_fict_, ex, ey, hz, nx, ny, tmax);

	gpu_timer_set();
	//GPU_argv_init();
	fdtdCuda(_fict_, ex, ey, hz, nx, ny, tmax);//, hz_outputFromGpu);
	cudaDeviceSynchronize();
	gpu_timer_pause();
	gpu_timer_record(argc, argv);

	//t_start = rtclock();
	//runFdtd(_fict_, ex, ey, hz);
	//t_end = rtclock();
	
	//fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);
	
	//compareResults(hz, hz_outputFromGpu);

	// FILE *fp;

	// fp = fopen("file.txt","w");

	// for(int i = 0; i < nx*ny; i+= 1000) {
	// 	fprintf(fp, "%lf\n", hz[i]);
	// }
	
	// fclose(fp);

	cudaFree(_fict_);
	cudaFree(ex);
	cudaFree(ey);
	cudaFree(hz);
	//free(hz_outputFromGpu);

	return 0;
}

