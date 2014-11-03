template <typename TYPE>
__global__ void bruteforce_distances (TYPE *A, TYPE *B , TYPE *D, int n, int m , int dim)
{

/*
 * - Input arguments:  Matrix A[dim][m] (Transposed form | with cuBLAS or low level kernel).
 *                     Matrix B[n][dim] (Original form).
 * - Output arguments: Matrix D [m][n] (Transposed form, but must be reshaped to original form after kernel execution).
 * 
 * - This kernel computes the distances D between the n dim-sized vectors of A and the m dim-sized vectors of B.  
 * - It has been tested on GPUs with cc 2.0, 2.1, 3.0 and 5.0 with TYPE = <float , int>
 * - To have a better performance in your platform, you will have to jimmy the CHUNK size.
 * 
 * Algorithms:
 * 1 - 
 */

#if (__CUDA_ARCH__ >= 500)      // Maxwell or newer
  #define CHUNK 16 
#elif (__CUDA_ARCH__ < 500)	// Kepler or older 
  #define CHUNK 4
#endif

extern __shared__ TYPE sB[];    // Allocation of shared mem @ compile time : < CHUNK * sizeof(TYPE) >

int bx  = blockIdx.x;		// Grid size  = < ceil(M/CHUNK) >
int tx  = threadIdx.x;          // Block size = < 128 >
  
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
      TYPE Atemp = A[(n*i)+tx];    // Reading elements of Matrix A for L1/L2 cache storage
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
