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


template <typename TYPE>
__global__ void bruteforce_distances (TYPE *A, TYPE *B , TYPE *D, int n, int m , int dim)
{

#if (__CUDA_ARCH__ >= 500)            // Maxwell or newer
  #define CHUNK 16 
#elif (__CUDA_ARCH__ < 500)	          // Kepler or older 
  #define CHUNK 4
#endif

extern __shared__ TYPE sB[];          // Allocation of shared mem @ runtime : < CHUNK * sizeof(TYPE) >

int bx  = blockIdx.x;		              // Grid size  = < ceil(M/CHUNK) >
int tx  = threadIdx.x;                // Block size = < dim >
  
for (int i = 0; i < CHUNK; i++)
    sB[(i*dim)+tx] = B[(((bx*CHUNK)+i)*dim)+tx];
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


// CUDA Kernel for transposing matrices with float elements
#define BLOCK_DIM 16

__global__ void transposing(float *odata, float *idata, int width, int height)
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

  float *h_A = new float[N * dim]; filling(h_A, N, dim, 100.0f);
  float *h_B = new float[M * dim]; filling(h_B, M, dim, 100.0f);
  float *h_D = new float[M * N];

  // Memory Allocation + GPU transferts
  float *d_A;   cudaMalloc((void**)&d_A , N * dim * sizeof(float));   // Matrix A
  float *d_AT;  cudaMalloc((void **)&d_AT , N * dim *sizeof(float));  // Transposed A
  float *d_B;   cudaMalloc((void**)&d_B , M * dim * sizeof(float));   // Matrix B
  float *d_D;   cudaMalloc((void**)&d_D , M * N * sizeof(float));     // Matrix D containing euclidean distances
  float *d_DT;  cudaMalloc((void**)&d_DT , M * N * sizeof(float));    // Transposed D

  cudaMemcpy(d_A , h_A , N * dim * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B , h_B , M * dim * sizeof(float), cudaMemcpyHostToDevice);

  // Transpose d_A on the GPU (d_AT)
  cudaFuncSetCacheConfig(transposing, cudaFuncCachePreferShared);

  dim3 blocks_t0  (ceil((float)dim / BLOCK_DIM), ceil((float)N / BLOCK_DIM));
  dim3 threads_t0 (BLOCK_DIM, BLOCK_DIM);
  transposing <<<blocks_t0,threads_t0>>> (d_AT , d_A , dim , N);
  cudaDeviceSynchronize();


  // KERNEL LAUNCH
    // Prefer L1 cache over shared memory
    cudaFuncSetCacheConfig(bruteforce_distances<float>, cudaFuncCachePreferL1);
    dim3 blocks  (ceil(M/CHUNK));
    dim3 threads (dim);
    bruteforce_distances <float> <<< blocks , threads , CHUNK * dim * sizeof(float)>>> (d_AT , d_B , d_DT , N , M , dim);
    cudaDeviceSynchronize();

  // Transpose d_DT on the GPU (d_D)
  dim3 blocks_t1  (ceil((float)N / BLOCK_DIM), ceil((float)M / BLOCK_DIM));
  dim3 threads_t1 (BLOCK_DIM, BLOCK_DIM);
  transposing <<<blocks_t1,threads_t1>>> (d_D , d_DT , N , M);
  cudaDeviceSynchronize();

  // Copying the results back to the host
  cudaMemcpy(h_D , d_D , N * M * sizeof(float) , cudaMemcpyDeviceToHost);

  // Freeing GPU data
  cudaFree(d_A);
  cudaFree(d_AT);
  cudaFree(d_B);
  cudaFree(d_DT);
  cudaFree(d_D);

  delete [] h_D;
  delete [] h_B;
  delete [] h_A;

  return 0;
}
