#include "../tools/checkerrors.h"
#include <iostream>

#define BLOCK_DIM 16	// Threadblock size for matrix transposition

template <typename TYPE>
__global__ void transposing(TYPE *odata, TYPE *idata, int width, int height)
{
	__shared__ float block[BLOCK_DIM][BLOCK_DIM+1];

	unsigned int xIndex = blockIdx.x * BLOCK_DIM + threadIdx.x;
	unsigned int yIndex = blockIdx.y * BLOCK_DIM + threadIdx.y;
	if((xIndex < width) && (yIndex < height))
	{
		unsigned int index_in = yIndex * width + xIndex;
		block[threadIdx.y][threadIdx.x] = idata[index_in];
	}
	__syncthreads();

	xIndex = blockIdx.y * BLOCK_DIM + threadIdx.x;
	yIndex = blockIdx.x * BLOCK_DIM + threadIdx.y;
	if((xIndex < height) && (yIndex < width))
	{
		unsigned int index_out = yIndex * height + xIndex;
		odata[index_out] = block[threadIdx.x][threadIdx.y];
	}
}


// Wrapper function
template <typename TYPE>
void call_transposing (TYPE *odata, TYPE *idata, int width, int height)
{
	  checkCuda(cudaFuncSetCacheConfig(transposing, cudaFuncCachePreferShared));

	  dim3 blocks  (ceil((float)width / BLOCK_DIM), ceil((float)height / BLOCK_DIM));
	  dim3 threads (BLOCK_DIM, BLOCK_DIM);
	  transposing <TYPE> <<<blocks,threads>>> (odata , idata , width , height);
	  checkCudaErrors();
}
