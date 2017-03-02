// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ...
// +
// lst[n-1]}

#include <wb.h>

#define BLOCK_SIZE 512 //@@ You can change this

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

__global__ void scan(float * input, float * output, int len) {

  __shared__ float T[BLOCK_SIZE*2];
 unsigned int t = threadIdx.x;
  unsigned int start = 2 * blockIdx.x * blockDim.x;

  //load subarray into the shared subarray vector
  if((t+start)>=len)
    input[t+start]=0;
  if((start+BLOCK_SIZE+t)>=len)
    input[start+t+BLOCK_SIZE]=0;
  T[t]=input[t+start];
  T[t+BLOCK_SIZE]=input[start+BLOCK_SIZE+t];
  __syncthreads(); //wait for load to finish.
    
  int stride = 1;
  while(stride<=BLOCK_SIZE)
  {
    int index = (threadIdx.x+1)*stride*2-1;
    if(index<BLOCK_SIZE*2)
    { T[index] += T[index-stride];}
    stride = stride * 2;
    __syncthreads();
  }
  stride = BLOCK_SIZE/2;
  while(stride>0)
  {
    int index = (threadIdx.x+1)*stride*2-1;
    if((index+stride)<BLOCK_SIZE*2)
    {
      T[index+stride] += T[index];
    }
    stride = stride/2;
    __syncthreads();
  }

  if ((t+start) < len)
    output[t+start] = T[t];
  if ((start+BLOCK_SIZE+t) < len)
    output[start+BLOCK_SIZE+t] = T[t + BLOCK_SIZE];
}


__global__ void arraysum(float * input, float * sum, int len){
  unsigned int t = threadIdx.x;
  //good idea to set this index. t represent the blockIdx, and index is the last item in each sharedmemory of each block.
  unsigned int index = (t + 1)*BLOCK_SIZE*2 - 1;
  if (index < len) {
    sum[t] = input[index];
  }
}

// in this kernel, the block size should be 2 * BLOCK_SIZE, because should match the former kernel block.
__global__ void add(float * input, float * inputsum, int len){
  int t = threadIdx.x;
  if(blockIdx.x<((len-1)/2*blockDim.x)){
    input[t+blockIdx.x*blockDim.x+blockDim.x]+= inputsum[blockIdx.x]; 
  }
}






int main(int argc, char **argv) {
  wbArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  float *deviceInput;
  float *deviceOutput;
  int numElements; // number of elements in the list
  float *arraysumlist;
  float *arraysumlistscan;

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &numElements);
  hostOutput = (float *)malloc(numElements * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The number of input elements in the input is ",
        numElements);

  wbTime_start(GPU, "Allocating GPU memory.");
  wbCheck(cudaMalloc((void **)&deviceInput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceOutput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&arraysumlist, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&arraysumlistscan, ceil(numElements/2*BLOCK_SIZE) * sizeof(float)));
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Clearing output memory.");
  wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float)));
  wbTime_stop(GPU, "Clearing output memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  wbCheck(cudaMemcpy(deviceInput, hostInput, numElements * sizeof(float),
                     cudaMemcpyHostToDevice));
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  dim3 DimGrid(ceil(numElements/1024.0), 1, 1);
  dim3 DimBlock(BLOCK_SIZE, 1 , 1);
  
   dim3 SaveGrid(1,1,1);
  dim3 SaveBlock(BLOCK_SIZE,1,1);
  //divide by extra factor of 2 because each subarray is BLOCK_SIZE*2 large.
  dim3 AccumGrid(1,1,1);
  dim3 AccumBlock(BLOCK_SIZE,1,1);
  dim3 AddBlock(BLOCK_SIZE*2,1,1);
  
  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Modify this to complete the functionality of the scan
  //@@ on the deivce
  scan<<<DimGrid , DimBlock>>>(deviceInput , deviceOutput ,numElements);
  
  arraysum<<<DimGrid,DimBlock>>>(deviceOutput,arraysumlist, numElements);
  //scan on the sumlist
  scan<<<DimGrid,DimBlock>>>(arraysumlist, arraysumlistscan, (numElements-1)/(BLOCK_SIZE*2)+1);
  //add sumlist to deviceOutput e.g. deviceOutput[idx] += sumList[idx/BLOCK_SIZE];
  add<<<DimGrid,AddBlock>>> (deviceOutput,arraysumlistscan, numElements);
  

  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");
  wbTime_start(Copy, "Copying output memory to the CPU");
  wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float),
                     cudaMemcpyDeviceToHost));
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostOutput, numElements);

  free(hostInput);
  free(hostOutput);

  return 0;
}

