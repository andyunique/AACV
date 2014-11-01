__global__ void gpuDistances(float *A, float *B , float *D, int n, int m , int dim)
{
#if (__CUDA_ARCH__ >= 500)      // On Maxwell architecture (cc5.0 or newer)
  #define CHUNK 16
#elif (__CUDA_ARCH__ < 500)
  #define CHUNK 4
#endif

extern __shared__ float sB[];       // Dynamic allocation
int bx  = blockIdx.x;

  int tx  = threadIdx.x;
  for (int i = 0; i < CHUNK; i++)
    sB[(i*dim)+tx] = B[(((bx*CHUNK)+i)*dim)+tx];
  __syncthreads();

  while (tx < n)
  {
    float result[CHUNK];
    for (int i = 0; i < CHUNK; i++)   // Initializing result array
      result[i] = 0.0f;

    for (int i = 0; i < dim; i++)
    {
      float Atemp = A[(n*i)+tx];    // Reading elements in array A for L2 cache storage
      for (int j = 0; j < CHUNK; j++)
      {
        float temp = Atemp - sB[i + (j*dim)];
        result[j] += temp * temp;
      }
    }
    for (int i = 0; i < CHUNK; i++)   // Copying final results to global memory
      D[((i+(bx*CHUNK))*n)+ tx] = result[i];
    tx += blockDim.x;
  }
}