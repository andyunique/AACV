class GpuTimer
{
private:
  cudaEvent_t tic;
  cudaEvent_t toc;

public:
  GpuTimer()
  {
    cudaEventCreate(&tic);
    cudaEventCreate(&toc);
  }

  ~GpuTimer()
  {
    cudaEventDestroy(tic);
    cudaEventDestroy(toc);
  }

  void start()
  {
    cudaEventRecord(tic, 0);
  }

  void stop()
  {
    cudaEventRecord(toc, 0);
  }

  float elapsedTime()
  {
    float elapsed;
    cudaEventSynchronize(toc);
    cudaEventElapsedTime(&elapsed, tic, toc);
    return elapsed;
  }
};
