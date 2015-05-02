/*
*   This program compare different CUDA kernels performing basic histogram over float linear array
*
*	histogram_atomics (Naive) : Using atomics at block + grid scale
*	histogram_reduce_explicit : Using reduction at block scale + atomics at grid scale
*	histogram_reduce_implicit : Using reduction with implicit warp synchronization at block scale + atomics at grid scale
*
*   For more details, please visit: http://sett.com/gpgpu/cuda-leveraging-implicit-intra-warp-synchronization-in-reduction-algorithms
*
* - FEEL FREE TO UPGRADE IT !
*/

#include "../tools/checkerrors.h"
#include "../tools/gputimer.h"


__global__ void histogram_atomics(const float *idata, int *ohist, const int dataSize, const int nbin)
{
	unsigned int tidxx = threadIdx.x;
	unsigned int gTidx = blockDim.x * blockIdx.x + threadIdx.x;

	if (gTidx < dataSize)
	{
		float item = idata[gTidx];
		int bin = ((int)item) % nbin;

		atomicAdd(&ohist[bin], 1);
	}
}

__global__ void histogram_reduce_explicit(const float *idata, int *ohist, const int dataSize, const int nbin)
{
	__shared__ int sh_data_tmp[40][32];

	unsigned int tidxx = threadIdx.x;
	unsigned int gTidx = blockDim.x * blockIdx.x + threadIdx.x;

	for (int i = 0; i < nbin; ++i)
		sh_data_tmp[i][tidxx] = 0;
	__syncthreads();

	if (gTidx < dataSize)
	{
		// filling the 32 local histograms
		float item = idata[gTidx];
		int bin = ((int)item) % nbin;
		sh_data_tmp[bin][tidxx] += 1;
	}
	__syncthreads();

	//Reduction with explicit warp synchronization
	if (tidxx < 16)
	{
		for (int k = 0; k < nbin; ++k)
		{
			sh_data_tmp[k][tidxx] += sh_data_tmp[k][tidxx + 16]; __syncthreads();
			sh_data_tmp[k][tidxx] += sh_data_tmp[k][tidxx + 8];  __syncthreads();
			sh_data_tmp[k][tidxx] += sh_data_tmp[k][tidxx + 4];  __syncthreads();
			sh_data_tmp[k][tidxx] += sh_data_tmp[k][tidxx + 2];  __syncthreads();
			sh_data_tmp[k][tidxx] += sh_data_tmp[k][tidxx + 1];  __syncthreads();

			if (tidxx == 0) // Updating Global Histogram using atomics at grid scale
				atomicAdd(&ohist[k], sh_data_tmp[k][0]);
		}
	}
}

__global__ void histogram_reduce_implicit(const float *idata, int *ohist, const int dataSize, const int nbin)
{

	__shared__ volatile int sh_data_tmp[40][32];

	unsigned int tidxx = threadIdx.x;
	unsigned int gTidx = blockDim.x * blockIdx.x + threadIdx.x;

	for (int i = 0; i < nbin; ++i)
		sh_data_tmp[i][tidxx] = 0;
	__syncthreads();

	if (gTidx < dataSize)
	{
		// filling the 32 local histograms
		float item = idata[gTidx];
		int bin = ((int)item) % nbin;
		sh_data_tmp[bin][tidxx] += 1;
	}
	__syncthreads();

	//Reduction with implicit warp synchronization
	if (tidxx < 16)
	{
		for (int k = 0; k < nbin; ++k)
		{
			sh_data_tmp[k][tidxx] += sh_data_tmp[k][tidxx + 16];
			sh_data_tmp[k][tidxx] += sh_data_tmp[k][tidxx + 8];
			sh_data_tmp[k][tidxx] += sh_data_tmp[k][tidxx + 4];
			sh_data_tmp[k][tidxx] += sh_data_tmp[k][tidxx + 2];
			sh_data_tmp[k][tidxx] += sh_data_tmp[k][tidxx + 1];

			if (tidxx == 0) // Updating Global Histogram using atomics at grid scale
				atomicAdd(&ohist[k], sh_data_tmp[k][0]);
		}
	}
}

int main()
{
	const int histSize = 40;
	const int dataSize = 2000000;
	GpuTimer timer;

	float *h_data = new float[dataSize];
	int *h_hist = new int[histSize];
	int *h_histi = new int[histSize];
	int *h_hist2 = new int[histSize];

	// Initializing data
	for (int i = 0; i < dataSize; i++)
		h_data[i] = 10.0f + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (1000.0f - 10.0f)));

	for (int i = 0; i < histSize; ++i) // Output of histogram_reduce_explicit
		h_hist[i] = 0;
	for (int i = 0; i < histSize; ++i) // Output of histogram_reduce_implicit
		h_histi[i] = 0;
	for (int i = 0; i < histSize; ++i) // Output of histogram_atomics
		h_hist2[i] = 0;

	float *d_data;
	int * d_hist;
	checkCuda(cudaMalloc((void**)&d_data, dataSize * sizeof(float)));
	checkCuda(cudaMalloc((void**)&d_hist, histSize * sizeof(int)));
	checkCuda(cudaMemcpy(d_data, h_data, dataSize * sizeof(float), cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpy(d_hist, h_hist, histSize* sizeof(int), cudaMemcpyHostToDevice));

	dim3 block(32);
	dim3 grid(ceil((float)(dataSize / 32)));

	checkCuda(cudaFuncSetCacheConfig(histogram_reduce_explicit, cudaFuncCachePreferShared));
	checkCuda(cudaFuncSetCacheConfig(histogram_reduce_implicit, cudaFuncCachePreferShared));

	timer.start();
	histogram_reduce_explicit << < grid, block >> > (d_data, d_hist, dataSize, histSize);
	checkCuda(cudaDeviceSynchronize());
	timer.stop();
	std::cout << "Elapsed time (reduce with explicit synchro): " << timer.elapsedTime() / 1000 << " ms" << std::endl;
	checkCuda(cudaMemcpy(h_hist, d_hist, histSize * sizeof(int), cudaMemcpyDeviceToHost));


	checkCuda(cudaMemcpy(d_hist, h_histi, histSize* sizeof(int), cudaMemcpyHostToDevice));
	timer.start();
	histogram_reduce_implicit << < grid, block >> > (d_data, d_hist, dataSize, histSize);
	checkCuda(cudaDeviceSynchronize());
	timer.stop();
	std::cout << "Elapsed time (reduce with implicit synchro): " << timer.elapsedTime() / 1000 << " ms" << std::endl;
	checkCuda(cudaMemcpy(h_histi, d_hist, histSize * sizeof(int), cudaMemcpyDeviceToHost));

	checkCuda(cudaMemcpy(d_hist, h_hist2, histSize* sizeof(int), cudaMemcpyHostToDevice));
	timer.start();
	histogram_atomics << < ceil((float)dataSize / 1024), 1024 >> > (d_data, d_hist, dataSize, histSize);
	checkCuda(cudaDeviceSynchronize());
	timer.stop();
	std::cout << "Elapsed time (atomics): " << timer.elapsedTime() / 1000 << " ms" << std::endl;
	checkCuda(cudaMemcpy(h_hist2, d_hist, histSize * sizeof(int), cudaMemcpyDeviceToHost));

	// Printing result
	for (int i = 0; i < histSize; ++i)
		std::cout << "hist value at " << i << " is : " << h_hist[i] << " | " << h_histi[i] << "& atomics-> " << h_hist2[i] << std::endl;


	system("pause");

	delete h_hist;
	h_hist = NULL;
	delete h_histi;
	h_histi = NULL;
	delete h_hist2;
	h_hist = NULL;
	delete h_data;
	h_data = NULL;

	checkCuda(cudaFree(d_data));
	checkCuda(cudaFree(d_hist));

	return 0;
}

