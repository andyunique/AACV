/*
 * - This kernel computes the Euclidean distances D between the n dim-sized vectors of A and the m dim-sized vectors of B.  
 *
 * - Input arguments:  Matrix A[dim][m] (Transposed form | with cuBLAS or low level kernel).
 *                     Matrix B[n][dim] (Original form).
 * - Output arguments: Matrix D [m][n]  (Transposed form | must be reshaped to original form after kernel execution).
 * 
 * 
 * - Kernel details:
 *    - CHUNK: Number of B's vectors stored in shared memory.
 *    - Block size = dim (vector's length)
 *    - grid size  = m / CHUNK
 *    - All threadblocks read A's transposed vectors to store them in L1/L2 caches (The transposed form allows coalesced global mem accesses).
 *    - Every threadblock computes CHUNK * n distances.
 *
 * - More details and advices:
 *    - It has been tested on GPUs with cc 2.0, 2.1, 3.0 and 5.0 with TYPE = <float , int>
 *    - To be tested with double type shortly to see whether the performance decreases since double variables occupy 2 4-bytes words in shared mem.
 *    - To have a better performance in your platform, you will have to jimmy the CHUNK size as its value has been chosen heuristically.
 *    - The more SM and L2 cache you have, the better it will perform.
 * 
 * - Some results with A[128][4100], B[24600][128]
 *    - On Fermi architecture (Quadro 1000m - GF108) : around 700ms
 *    - On Kepler architecture (GTX 650 - GK107) :  around 400 ms
 *    - On Maxwell architecture (GTX 750 Ti - GM107) : around 90 ms
 *    
 * - FEEL FREE TO UPGRADE IT !
 */

#include "../tools/checkerrors.h"
#include "../tools/wrappers.h"

template <typename TYPE>
__global__ void bruteforce_distances (TYPE *A, TYPE *B , TYPE *D, int n, int m , int dim)
{

#if (__CUDA_ARCH__ >= 500)            // Maxwell or newer
  #define CHUNK 16 
#elif (__CUDA_ARCH__ < 500)	          // Kepler or older 
  #define CHUNK 4
#endif

extern __shared__ TYPE sB[];          // Allocation of shared mem @ runtime : < CHUNK * sizeof(TYPE) >

int bx  = blockIdx.x;		          // Grid size  = < ceil(M/CHUNK) >
int tx  = threadIdx.x;                // Block size = < dim >
  
for (int i = 0; i < CHUNK; i++)
  sB[(i*dim)+threadIdx.x] = B[(((bx*CHUNK)+i)*dim)+tx];
  __syncthreads();

  while (tx < n)
  {
    TYPE result[CHUNK];
    
    for (int i = 0; i < CHUNK; i++)   // Initializing result array
      result[i] = (TYPE) 0.0f;

    for (int i = 0; i < dim; i++)
    {
      TYPE Atemp = A[(n*i)+tx];       // Reading elements of Matrix A for L1/L2 cache storage
      
      for (int j = 0; j < CHUNK; j++)
      {
        TYPE temp = Atemp - sB[i + (j*dim)];
        result[j] += temp * temp;
      }
    }
   
    for (int i = 0; i < CHUNK; i++)   // Copying final results to global memory
      D[((i+(bx*CHUNK))*n)+ tx] = result[i];
    tx += blockDim.x;
  }
}

// Function for filling input matrices with random values
#include <cfloat>
void filling (float *matrix, int width , int height, float high) // high = highest value
{
  float low = 0.0f;
  for(int i = 0; i < width * height; i++)
    matrix[i]= low + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(high - low)));
}

// MAIN : EXAMPLE OF KERNEL LAUNCH (using float elements)

int main (int argc , char *argv[])
{
  // Initializing host data ... (N, M, dim, h_A , h_B , h_D, etc...)
  const int dim = 128;
  const int N = 4100;
  const int M = 24600;

  // Host memory allocation -- page-locked for faster transfers
  float *h_A; checkCuda(cudaMallocHost((void**)&h_A, N * dim * sizeof(float))); 
  filling(h_A, N, dim, 100.0f);
  float *h_B; checkCuda(cudaMallocHost((void**)&h_B, M * dim * sizeof(float))); 
  filling(h_B, M, dim, 100.0f);
  float *h_D; checkCuda(cudaMallocHost((void**)&h_D, M * N * sizeof(float))); 

  // Memory Allocation + GPU transferts
  float *d_A;   checkCuda(cudaMalloc((void**)&d_A , N * dim * sizeof(float)));  // Matrix A
  float *d_AT;  checkCuda(cudaMalloc((void **)&d_AT , N * dim *sizeof(float))); // Transposed A
  float *d_B;   checkCuda(cudaMalloc((void**)&d_B , M * dim * sizeof(float)));  // Matrix B
  float *d_D;   checkCuda(cudaMalloc((void**)&d_D , M * N * sizeof(float)));    // Matrix D containing euclidean distances
  float *d_DT;  checkCuda(cudaMalloc((void**)&d_DT , M * N * sizeof(float)));   // Transposed D

  checkCuda(cudaMemcpy(d_A , h_A , N * dim * sizeof(float), cudaMemcpyHostToDevice));
  checkCuda(cudaMemcpy(d_B , h_B , M * dim * sizeof(float), cudaMemcpyHostToDevice));

  call_transposing <float> (d_AT , d_A , dim , N); // Transposing A

	  // KERNEL LAUNCH
	  // Prefer L1 cache over shared memory
	  checkCuda(cudaFuncSetCacheConfig(bruteforce_distances<float>, cudaFuncCachePreferL1));
	  dim3 blocks  (ceil(M/CHUNK));
	  dim3 threads (dim);
	  int smemBytes = CHUNK * dim * sizeof(float);
	  bruteforce_distances <float> <<< blocks , threads , smemBytes>>> (d_AT , d_B , d_DT , N , M , dim);
	  checkCudaErrors();

	 call_transposing <float> (d_D , d_DT , N , M);	// Transposing back D

  // Copying the results back to the host
  checkCuda(cudaMemcpy(h_D , d_D , N * M * sizeof(float) , cudaMemcpyDeviceToHost));

  // Freeing GPU data
  checkCuda(cudaFree(d_A));
  checkCuda(cudaFree(d_AT));
  checkCuda(cudaFree(d_B));
  checkCuda(cudaFree(d_DT));
  checkCuda(cudaFree(d_D));
  checkCuda(cudaFree(h_D));
  checkCuda(cudaFree(h_B));
  checkCuda(cudaFree(h_A));

  checkCuda(cudaDeviceReset());

  return 0;
}
