#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"

#include "../tools/checkerrors.h"
#include "../tools/gputimer.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <iomanip>

/*
These 2 Kernels computes the projection of a mesh for a given camera position in the 3D space.

Since the program correctness relies heavily on single-float precision, we recommend to disable the Floating-Point Multiply-Add (fmad) option.
The CUDA compiler (nvcc) is configured to produce FMAD instructions by default. To request it to stop producing
FMAD instructions and use the normal floating-point instructions use the compiler directive --fmad=false.
By doing so, the multiply-add operations are no longer in compliance with the IEEE 754 rule. 
The results becomes more accurate with however a little performance hit.

*/

__constant__ int width;
__constant__ int height;
__constant__ int nVertices;
__constant__ float extrinsic[16];
__constant__ float intrinsic[9];
__constant__ int d_size;

__device__ float4 _3Dto2D (const float4 vec)
{
	float4 res;
	res.x = intrinsic[0] * vec.x + intrinsic[2] * vec.z;
	res.y = intrinsic[4] * vec.y + intrinsic[5] * vec.z;
	res.z = vec.z;
	res.w = 0.0f;
	return res;
}


// Converting from world coordinates to camera coordinates
__device__ float4 worldToCamera (const float4 vec)
{
	float4 res;
	res.x = extrinsic[0] * vec.x + extrinsic[1] * vec.y + extrinsic[2] * vec.z + extrinsic[3] * vec.w;
	res.y = extrinsic[4] * vec.x + extrinsic[5] * vec.y + extrinsic[6] * vec.z + extrinsic[7] * vec.w;
	res.z = extrinsic[8] * vec.x + extrinsic[9] * vec.y + extrinsic[10] * vec.z + extrinsic[11] * vec.w;
	res.w = extrinsic[12] * vec.x + extrinsic[13] * vec.y + extrinsic[14] * vec.z + extrinsic[15] * vec.w;

	return res;
}

// Customized Atomic operation supporting floats
__device__ float fatomicMin(volatile float* address, float val)
{
	int* address_as_i = (int*)address;
	int old = *address_as_i, assumed;
	do {
		assumed = old;
		old = atomicCAS(address_as_i, assumed,
			__float_as_int(fminf(val, __int_as_float(assumed))));
	} while (assumed != old);

	return __int_as_float(old);
}


__global__ void ComputeShortestDistances(const float* __restrict__ iVertices, volatile float* distancesBuffer)
{
	extern __shared__ float sh_vertices_tmp[];

	int bdimx = blockDim.x;
	int bidxx = blockIdx.x;
	int tidxx = threadIdx.x;
	int gTidx = bidxx * bdimx + tidxx;

	if (gTidx >= nVertices) return;


#pragma unroll
	for (int iter = 0; iter < 3; ++iter)
		sh_vertices_tmp[iter * bdimx + tidxx] = iVertices[iter * nVertices + gTidx];
	__syncthreads();

	float4 vertex;
	vertex.x = sh_vertices_tmp[tidxx];
	vertex.y = sh_vertices_tmp[bdimx + tidxx];
	vertex.z = sh_vertices_tmp[2 * bdimx + tidxx];
	vertex.w = 1.0f;

	float4 pixel = worldToCamera(vertex);

	float ww = 1.0f / pixel.w;
	pixel.x = pixel.x * ww;
	pixel.y = pixel.y * ww;
	pixel.z = pixel.z * ww;
	pixel.w = 1.0f;

	float distance = pixel.x * pixel.x + pixel.y * pixel.y + pixel.z * pixel.z;

	pixel = _3Dto2D(pixel);

	float zz = 1.0f / pixel.z;
	int x = (int)pixel.x * zz;
	int y = (int)pixel.y * zz;

	if (x >= 0 && y >= 0 && x < width && y < height)
	{
		fatomicMin(&distancesBuffer[y*width + x], distance);
	}
}


__global__ void SelectNearestVertices(const float* __restrict__ iVertices, const float* __restrict__ distancesBuffer, float * __restrict__ oProjected_mesh_vertices)
{
	extern __shared__ float sh_vertices_tmp[];

	int bdimx = blockDim.x;
	int bidxx = blockIdx.x;
	int tidxx = threadIdx.x;
	int gTidx = bidxx * bdimx + tidxx;

	if (gTidx >= nVertices) return;

#pragma unroll
	for (int iter = 0; iter < 6; ++iter)
		sh_vertices_tmp[iter * bdimx + tidxx] = iVertices[iter * nVertices + gTidx];
	__syncthreads();

	float4 vertex;
	float4 normal;
	vertex.x = sh_vertices_tmp[tidxx];
	vertex.y = sh_vertices_tmp[bdimx + tidxx];
	vertex.z = sh_vertices_tmp[2 * bdimx + tidxx];
	vertex.w = 1.0f;
	normal.x = sh_vertices_tmp[3 * bdimx + tidxx];
	normal.y = sh_vertices_tmp[4 * bdimx + tidxx];
	normal.z = sh_vertices_tmp[5 * bdimx + tidxx];
	normal.w = 1.0f;

	float4 pixel = worldToCamera(vertex);

	float ww = 1.0f / pixel.w;
	pixel.x = pixel.x * ww;
	pixel.y = pixel.y * ww;
	pixel.z = pixel.z * ww;
	pixel.w = 1.0f;

	float distance = pixel.x * pixel.x + pixel.y * pixel.y + pixel.z * pixel.z;

	pixel = _3Dto2D(pixel);

	float zz = 1.0f / pixel.z;
	int x = (int)pixel.x * zz;
	int y = (int)pixel.y * zz;

	if (x >= 0 && y >= 0 && x < width && y < height)
	{
		if (distance == distancesBuffer[y*width + x])
		{
			oProjected_mesh_vertices[y*width + x] = vertex.x;
			oProjected_mesh_vertices[d_size + y*width + x] = vertex.y;
			oProjected_mesh_vertices[2 * d_size + y*width + x] = vertex.z;

			oProjected_mesh_vertices[3 * d_size + y*width + x] = normal.x;
			oProjected_mesh_vertices[4 * d_size + y*width + x] = normal.y;
			oProjected_mesh_vertices[5 * d_size + y*width + x] = normal.z;
		}
	}
}


int computeVerticesNb(const std::string& file)
{
	int cpt = 0;
	std::ifstream meshFile(file);
	std::string line;
	while (!meshFile.eof())
	{
		std::getline(meshFile, line);
		if (line[0] == 'v' && line[1] == ' ') ++cpt;
	}

	return cpt;
}

int extractMeshData(const std::string& file, float *odata, const int& verticesNb)
{
	std::ifstream meshFile(file);
	std::string line;
	std::stringstream stream;
	int vcpt = 0;
	int vncpt = 0;
	char junk[2];

	while (!meshFile.eof())
	{
		std::getline(meshFile, line);
		if (line.empty()) continue;

		if (line[0] == 'v' && line[1] == ' ')
		{
			stream << line;
			stream >> junk[0]
				>> odata[vcpt]
				>> odata[verticesNb + vcpt]
				>> odata[2 * verticesNb + vcpt];
			++vcpt;
		}
		else if (line[0] == 'v' && line[1] == 'n')
		{
			stream << line;
			stream >> junk[0] >> junk[1]
				>> odata[3 * verticesNb + vncpt]
				>> odata[4 * verticesNb + vncpt]
				>> odata[5 * verticesNb + vncpt];

			++vncpt;
		}

		stream.str(std::string());
		stream.clear();
	}

	return 1;
}

int printMeshProjection(const std::string& path, const int& stride, const float *data, const int &mode = 1)
{
	std::ofstream result;
	result.open(path.c_str());

	if (mode == 1)
	{
		for (int i = 0; i < stride; ++i) // printing projection results
		{
			if (data[i] == 0.0f && data[stride + i] == 0.0f && data[2 * stride + i] == 0.0f
				&& data[3 * stride + i] == 0.0f && data[4 * stride + i] == 0.0f && data[5 * stride + i] == 0.0f)
				continue;

			result << "v " << data[i] << " "
				<< data[stride + i] << " "
				<< data[2 * stride + i] << std::endl
				<< "vn " << data[3 * stride + i] << " "
				<< data[4 * stride + i] << " "
				<< data[5 * stride + i] << std::endl;
		}
	}
	else if (mode == 2) // checking mesh reading
	{
		for (int i = 0; i < stride; ++i)
		{
			result << "v " << data[i] << " "
				<< data[stride + i] << " "
				<< data[2 * stride + i] << std::endl;
		}
		for (int i = 0; i < stride; ++i)
		{
			result << "vn " << data[3 * stride + i] << " "
				<< data[4 * stride + i] << " "
				<< data[5 * stride + i] << std::endl;
		}
	}

	result.close();
	return 1;
}



int main(int argc, char **argv)
{
	float exMatrix[16] = { 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f };
	float inMatrix[9] = { 500.0f, 0.0f, 320.0f, 0.0f, 500.0f, 240.0f, 0.0f, 0.0f, 1.0f };
	int h_width = 640;
	int h_height = 480;
	int size = h_height * h_width;
	GpuTimer gpu_timer;

	std::string meshFilename    = "C:\\path\\to\\your\\input\\mesh.obj";
	std::string projectionFilename = "C:\\path\\to\\your\\output\\meshProjection.obj";

	int verticesNb = computeVerticesNb(meshFilename);

	float *h_vertices = new float[6 * verticesNb];
	std::fill(&h_vertices[0], &h_vertices[6 * verticesNb], 0.0f);
	std::cout << "Extracting data from Mesh with " << verticesNb << " vertices ..." << std::endl;
	int rc = extractMeshData(meshFilename, h_vertices, verticesNb);
	if (rc == 1) std::cout << "Successful mesh reading" << std::endl;

	float *h_distanceBuffer = new float[h_height * h_width];
	std::fill(&h_distanceBuffer[0], &h_distanceBuffer[h_height * h_width], FLT_MAX);

	float *h_projectedMeshVertices = new float[6 * h_height * h_width];
	std::fill(&h_projectedMeshVertices[0], &h_projectedMeshVertices[6 * h_height * h_width], 0.0f);

	checkCuda(cudaMemcpyToSymbol(width, &h_width, sizeof(int)));
	checkCuda(cudaMemcpyToSymbol(height, &h_height, sizeof(int)));
	checkCuda(cudaMemcpyToSymbol(nVertices, &verticesNb, sizeof(int)));
	checkCuda(cudaMemcpyToSymbol(extrinsic, &exMatrix, 16 * sizeof(float)));
	checkCuda(cudaMemcpyToSymbol(intrinsic, &inMatrix, 9 * sizeof(float)));
	checkCuda(cudaMemcpyToSymbol(d_size, &size, sizeof(int)));


	float *d_vertices = NULL;
	checkCuda(cudaMalloc((void**)&d_vertices, 6 * verticesNb * sizeof(float)));
	float *d_distanceBuffer = NULL;
	checkCuda(cudaMalloc((void**)&d_distanceBuffer, h_width * h_height * sizeof(float)));
	float *d_projectedMeshVertices = NULL;
	checkCuda(cudaMalloc((void**)&d_projectedMeshVertices, 6 * h_width * h_height * sizeof(float)));

	dim3 block(32);
	dim3 grid(ceil((float)verticesNb / block.x));

	checkCuda(cudaMemcpy(d_vertices, h_vertices, 6 * verticesNb * sizeof(float), cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpy(d_distanceBuffer, h_distanceBuffer, h_height * h_width * sizeof(float), cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpy(d_projectedMeshVertices, h_projectedMeshVertices, 6 * h_height * h_width * sizeof(float), cudaMemcpyHostToDevice));

	// Kernel 1
	gpu_timer.start();
	ComputeShortestDistances <<< grid, block, 3 * block.x  * sizeof(float) >>> (d_vertices, d_distanceBuffer);
	checkCuda(cudaDeviceSynchronize());
	gpu_timer.stop();
	std::cout << "Elapsed GPU time (ComputeShortestDistances ) = " << gpu_timer.elapsedTime() << " ms" << std::endl;

	// Kernel 2
	gpu_timer.start();
	SelectNearestVertices <<< grid, block, 6 * block.x  * sizeof(float) >>> (d_vertices, d_distanceBuffer, d_projectedMeshVertices);
	checkCuda(cudaDeviceSynchronize());
	gpu_timer.stop();
	std::cout << "Elapsed GPU time (SelectNearestVertices) = " << gpu_timer.elapsedTime() << " ms" << std::endl;

	checkCuda(cudaMemcpy(h_projectedMeshVertices, d_projectedMeshVertices, 6 * h_height * h_width * sizeof(float), cudaMemcpyDeviceToHost));
	rc = printMeshProjection(projectionFilename, size, h_projectedMeshVertices);
	if (rc == 1) std::cout << "GPU Mesh projection printed successfully." << std::endl;

	if (h_vertices)
	{
		delete h_vertices;
		h_vertices = NULL;
	}
	if (h_distanceBuffer)
	{
		delete h_distanceBuffer;
		h_distanceBuffer = NULL;
	}

	checkCuda(cudaFree(d_vertices));
	checkCuda(cudaFree(d_distanceBuffer));
	checkCuda(cudaFree(d_projectedMeshVertices));

	checkCuda(cudaDeviceReset());

#ifdef _WIN32
	system("pause");
#endif


	return 0;
}

