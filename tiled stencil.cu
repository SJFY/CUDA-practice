#include <wb.h>

#define wbCheck(stmt)                                                                         \
  do {                                                                                        \
    cudaError_t err = stmt;                                                                   \
    if (err != cudaSuccess) {                                                                 \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                                             \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));                          \
      return -1;                                                                              \
    }                                                                                         \
  } while (0)

__host__ __device__ float Clamp(float val, float start, float end) {
  return max(min(val, end), start);
}
#define Tilewidth 32
void stencil_cpu(float *_out, float *_in, int width, int height,
                         int depth) {
// (i,height) is used for z-axis
// (j,width) is used for y-axis
// (k,depth) is used for x-axis

#define out(i, j, k) _out[((i)*width + (j)) * depth + (k)]
#define in(i, j, k) _in[((i)*width + (j)) * depth + (k)]

  float res;
  for (int i = 1; i < height - 1; ++i) {
    for (int j = 1; j < width - 1; ++j) {
      for (int k = 1; k < depth - 1; ++k) {
        res = in(i, j, k + 1) + in(i, j, k - 1) + in(i, j + 1, k) +
              in(i, j - 1, k) + in(i + 1, j, k) + in(i - 1, j, k) -
              6 * in(i, j, k);
        out(i, j, k) = Clamp(res, 0, 255);
      }
    }
  }
#undef out
#undef in
}

__global__ void stencil(float *output, float *input, int width, int height,
                        int depth) {
  //@@ INSERT CODE HERE
  #define output(i, j, k) output[((i)*width + (j)) * depth + (k)]
  #define input(i, j, k) input[((i)*width + (j)) * depth + (k)]
  
  
  //i is z axis(height),j is y axis(width), k is x axis (depth)
  
  __shared__ float shared_array[Tilewidth][Tilewidth];
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int idz = threadIdx.y + blockIdx.y * blockDim.y;
  if((idx<width)&&(idz<depth)){
    float bottom = input(0,idx,idz);
    float current = input(1,idx,idz);
    float top = input(2,idx,idz);
    shared_array[threadIdx.y][threadIdx.x] = current;
    float result = 0;
    float x0,x2,y0,y2;
    __syncthreads();
    for(int k = 1; k < height-1 ; k++){
     /* if(idx==0 || idx== depth-1 ||  idy== width-1 || idy== 0){
       shared_array[threadIdx.y][threadIdx.x] = input(k,idy,idx);
        __syncthreads();
      }*/
    /*  if((idx ==0 && idz ==0)||(idx==width-1 && idz == 0)||(idx==0 && idz==depth-1)||(idx==width-1 && idz ==depth -1)){
       //  shared_array[threadIdx.y][threadIdx.x] = input(k,idx,idz);
        
      }
      else if ((idx > width -1)||(idz>depth -1 )){
        
      }
      else{*/
    
      x0 = (threadIdx.x>0)?shared_array[threadIdx.y][threadIdx.x-1]:(idx==0)?0:input(k,idx-1,idz);
      x2 = (threadIdx.x<blockDim.x-1)?shared_array[threadIdx.y][threadIdx.x+1]:(idx==width-1)?0:input(k,idx+1,idz);
      y0 = (threadIdx.y>0)?shared_array[threadIdx.y-1][threadIdx.x]:(idz==0)?0:input(k,idx,idz-1);
      y2 = (threadIdx.y<blockDim.y-1)?shared_array[threadIdx.y+1][threadIdx.x]:(idz==depth-1)?0:input(k,idx,idz+1);
      result = top + bottom + x0 + x2 + y0 + y2 - 6 * current;
      __syncthreads();
      bottom = current;
      current = top; 
      shared_array[threadIdx.y][threadIdx.x] = current;
        if((k+1)<height-1)
      top = input(k+2,idx,idz);
      __syncthreads();
      if((idx>0 && idx < (width -1))&&(idz>0 && idz<(depth -1)))
      output(k,idx,idz) = Clamp(result,0,255);
     // }
      }    
  #undef output
  #undef input
    }
}
  
static void launch_stencil(float *deviceOutputData, float *deviceInputData, int width,
                           int height, int depth) {
  //@@ INSERT CODE HERE
  dim3 DimGrid(ceil((width-1)/32.0),ceil((depth-1)/32.0),1);
			dim3 DimBlock(32,32,1);
		
			stencil<<<DimGrid,DimBlock>>>(deviceOutputData,deviceInputData,width,height,depth);
}

int main(int argc, char *argv[]) {
  wbArg_t arg;
  int width;
  int height;
  int depth;
  char *inputFile;
  wbImage_t input;
  wbImage_t output;
  float *hostInputData;
  float *hostOutputData;
  float *deviceInputData;
  float *deviceOutputData;

  arg = wbArg_read(argc, argv);

  inputFile = wbArg_getInputFile(arg, 0);

  input = wbImport(inputFile);

  height = wbImage_getHeight(input);
  width = wbImage_getWidth(input);
  depth = wbImage_getChannels(input);

  output = wbImage_new(width,height,depth);

  hostInputData = wbImage_getData(input);
  hostOutputData = wbImage_getData(output);

  wbTime_start(GPU, "Doing GPU memory allocation");
  cudaMalloc((void **)&deviceInputData, width * height * depth * sizeof(float));
  cudaMalloc((void **)&deviceOutputData, width * height * depth * sizeof(float));
  wbTime_stop(GPU, "Doing GPU memory allocation");

  wbTime_start(Copy, "Copying data to the GPU");
  cudaMemcpy(deviceInputData, hostInputData, width * height * depth * sizeof(float),
             cudaMemcpyHostToDevice);
  wbTime_stop(Copy, "Copying data to the GPU");

  wbTime_start(Compute, "Doing the computation on the GPU");
  launch_stencil(deviceOutputData, deviceInputData, width, height, depth);
  wbTime_stop(Compute, "Doing the computation on the GPU");

  wbTime_start(Copy, "Copying data from the GPU");
  cudaMemcpy(hostOutputData, deviceOutputData, width * height * depth * sizeof(float),
             cudaMemcpyDeviceToHost);
  wbTime_stop(Copy, "Copying data from the GPU");

  wbSolution(arg, output);

  cudaFree(deviceInputData);
  cudaFree(deviceOutputData);

  wbImage_delete(output);
  wbImage_delete(input);

  return 0;
}

