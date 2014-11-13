/***************** MAXIMUM SINGLE-SELL PROFIT **************/

// Serial version: Dynamic programming
float max_profit_serial( const float *data, const size_t num_prices)
{
float profit = 0.0f;
float cheapest = data[0];
int i;
for(i=0 ; i<num_prices ; i++)
{
	cheapest = std::min(cheapest , data[i]);
	profit   = std::max(profit , data[i] - cheapest); // Updating the max profit
}
return profit;
}


// Muli-threaded version : OpenMP
float max_profit_omp (const float * const data, const int num_prices , const int nthreads)
{
  float maxes[nthreads];
  float mins [nthreads];
  float profit = 0.0f;

  for (int i=0; i<nthreads; i++)
  {
    mins [i] = FLT_MAX;
    maxes[i] = FLT_MIN;
  }

  /* 
   * 1- Iterations of the parallel loop will be distributed in equal sized blocks
   *    to each thread in the team [schedule static].
   * 2- maxes and mins are shared among all threads.
   */
  #pragma omp parallel num_threads(nthreads) default(none) shared(mins, maxes) reduction(max:profit)
  {
      int index = omp_get_thread_num();
      float min = FLT_MAX, max = FLT_MIN;

      #pragma omp for schedule(static)	// In compile time
      for(int i=0; i < num_prices; i++)
      {
        if (data[i] < min) min=data[i];
        if (data[i] > max) max=data[i];
        if ((data[i] - min) > profit) profit = data[i] - min;
      }
      mins [index] = min;
      maxes[index] = max;
  }

  for (int i=0; i < nthreads-1; i++)
    for (int j=i+1; j < nthreads; j++)
        if ((maxes[j] - mins[i]) > profit) profit = maxes[j]-mins[i];

  return profit;
}


