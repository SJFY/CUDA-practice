// Histogram Equalization

#include <wb.h>

#define HISTOGRAM_LENGTH 256
#define Blocksize 512
#define BLOCK_SIZE 512

//@@ insert code here
#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

__global__ void cast(float * inputimage, unsigned char * ucharimage, int len){
  unsigned int t = threadIdx.x;
  unsigned int index = t+blockIdx.x * blockDim.x;
//  int stride = blockDim.x * gridDim.x;
  if(index<len){
    ucharimage[index]=(unsigned char)(255 * inputimage[index]);
   // index = index+stride;
  }
}

__global__ void convert(unsigned char * ucharimage, unsigned char * grayscaleimage, int graylen){
  unsigned int t = threadIdx.x;
  unsigned int index = t+blockIdx.x * blockDim.x;
//  int stride = blockDim.x * gridDim.x;
  if(index<graylen){
    grayscaleimage[index]=0.21*ucharimage[index*3]+0.71*ucharimage[index*3+1]+0.07*ucharimage[index*3+2];
  //  index = index + stride;
  }
}

__global__ void histogram(unsigned char * grayscaleimage, int graylen, unsigned int * histo){
  __shared__ unsigned int histo_private[256];
  if(threadIdx.x<256){
    histo_private[threadIdx.x] = 0;
  }
  __syncthreads();
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;
  while(i<graylen){
    atomicAdd(&(histo_private[grayscaleimage[i]]),1);
    i = i+stride;
  }
  __syncthreads();
  if(threadIdx.x<256){
    atomicAdd(&(histo[threadIdx.x]),histo_private[threadIdx.x]);
    
  }
}
/*
__global__ void histogram(unsigned char * grayscaleimage, int graylen, unsigned int * histo){
  

  __syncthreads();
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;
  while(i<graylen){
    atomicAdd(&(histo[grayscaleimage[i]]),1);
    i = i+stride;
  }
 
}
*/
/*
__global__ void scan(unsigned int * histo, float * CDF, int graphsize){
  int index = threadIdx.x + blockIdx.x*blockDim.x;
  if(index<256){
    histo[index]=histo[index]/graphsize;
  }
   __shared__ float T[BLOCK_SIZE*2];
 unsigned int t = threadIdx.x;
  unsigned int start = 2 * blockIdx.x * blockDim.x;

  //load subarray into the shared subarray vector
  if((t+start)>=256)
    histo[t+start]=0;
  if((start+BLOCK_SIZE+t)>=256)
    histo[start+t+BLOCK_SIZE]=0;
  T[t]=histo[t+start];
  T[t+BLOCK_SIZE]=histo[start+BLOCK_SIZE+t];
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

  if ((t+start) < 256)
    CDF[t+start] = T[t];
  if ((start+BLOCK_SIZE+t) < 256)
    CDF[start+BLOCK_SIZE+t] = T[t + BLOCK_SIZE];
}
*/

__global__ void correct(float * CDF, unsigned char * ucharimage, int len){
  int index = threadIdx.x+blockIdx.x*blockDim.x;
  if(index<len){
    unsigned int clamp = 255*(CDF[ucharimage[index]] - CDF[0])/(1 - CDF[0]);
    if(clamp>255){
      clamp = 255;
    }
    else if(clamp<0){
      clamp = 0;
    }
    else{
      clamp = clamp;
    }
    ucharimage[index] = clamp;
  }
}

__global__ void backtofloat(unsigned char * ucharimage, float * output, int len){
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if(index<len){
    output[index]=(float)(ucharimage[index]/255.0);
  }
}
//above is my code

int main(int argc, char **argv) {
  wbArg_t args;
  int imageWidth;
  int imageHeight;
  int imageChannels;
  wbImage_t inputImage;
  wbImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  const char *inputImageFile;

  //@@ Insert more code here
  float *deviceInputImageData;
  float *deviceOutputImageData;
  unsigned char *devicecharImageData;
  unsigned char *devicegrayImageData;
  unsigned int *histo;
  float *CDF;
  
  unsigned char * test;
  unsigned char * kerneloutput;
  unsigned char * kernelchar;
  unsigned int * kernelhisto;
  

  //above is my ocde

  args = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(args, 0);

  wbTime_start(Generic, "Importing data and creating memory on host");
  inputImage = wbImport(inputImageFile);
  imageWidth = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);
  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
  wbTime_stop(Generic, "Importing data and creating memory on host");

  //@@ insert code here
  hostInputImageData = wbImage_getData(inputImage) ;
//  hostOutputImageData = (unsigned char *)malloc(imageWidth*imageHeight*imageChannels * sizeof(unsigned char));
  hostOutputImageData = wbImage_getData(outputImage);
  test = (unsigned char *)malloc(imageWidth*imageHeight*imageChannels * sizeof(unsigned char));
  kerneloutput = (unsigned char *)malloc(imageWidth*imageHeight * sizeof(unsigned char));
  kernelchar = (unsigned char *)malloc(imageWidth*imageHeight*imageChannels * sizeof(unsigned char));
  kernelhisto = (unsigned int *)malloc(256 * sizeof(unsigned int));
  unsigned int* serial_histo = (unsigned int*) malloc(256 * sizeof(unsigned int));
  float* serialcdf = (float*) malloc(256*sizeof(float));
  float* kernelequcdf = (float*) malloc(256*sizeof(float));
  
  wbTime_start(GPU, "Allocating GPU memory.");
  wbCheck(cudaMalloc((void **)&deviceInputImageData, imageWidth*imageHeight*imageChannels * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceOutputImageData, imageWidth*imageHeight*imageChannels * sizeof(float)));
  wbCheck(cudaMalloc((void **)&devicecharImageData, imageWidth*imageHeight*imageChannels * sizeof(unsigned char)));
  wbCheck(cudaMalloc((void **)&devicegrayImageData, imageWidth*imageHeight* sizeof(unsigned char)));
  wbCheck(cudaMalloc((void **)&histo, 256* sizeof(unsigned int)));
  wbCheck(cudaMalloc((void **)&CDF, 256* sizeof(float)));
  wbTime_stop(GPU, "Allocating GPU memory.");
  
   wbTime_start(GPU, "Copying input memory to the GPU.");
  wbCheck(cudaMemcpy(deviceInputImageData, hostInputImageData, imageWidth*imageHeight*imageChannels * sizeof(float),
                     cudaMemcpyHostToDevice));
  wbTime_stop(GPU, "Copying input memory to the GPU.");
  
  dim3 DimGrid(ceil(imageWidth*imageHeight*imageChannels/512.0), 1, 1);
  dim3 DimBlock(Blocksize, 1 , 1);
  
  cast<<<DimGrid , DimBlock>>>(deviceInputImageData, devicecharImageData, imageWidth*imageHeight*imageChannels);
   
  dim3 DimGridconvert(ceil(imageWidth*imageHeight/512.0),1,1);
  convert<<<DimGridconvert , DimBlock>>>(devicecharImageData, devicegrayImageData, imageWidth*imageHeight);
  
   //test cast
    wbCheck(cudaMemcpy(kerneloutput, devicegrayImageData, imageWidth*imageHeight * sizeof(unsigned char),
                     cudaMemcpyDeviceToHost));
    wbCheck(cudaMemcpy(kernelchar, devicecharImageData, imageWidth*imageHeight*imageChannels * sizeof(unsigned char),
                     cudaMemcpyDeviceToHost));
  
    for(int i = 0; i<imageWidth*imageHeight;i++ ){
    test[i]= 0.21*kernelchar[3*i]+0.71*kernelchar[3*i+1]+0.07*kernelchar[3*i+2];
  }
 
  int faultnum = 0;
  for(int i = 0; i<imageWidth*imageHeight;i++ ){
    if(kerneloutput[i]== 68){
      faultnum ++;
      printf("%d  ",i);
    }
  }
  printf("%d/n total num %d/n",faultnum,imageWidth*imageHeight);
  // this part is good too
  //i don't need to copy test to grayimage
 // wbCheck(cudaMemcpy(devicegrayImageData, test, imageWidth*imageHeight * sizeof(unsigned char),
       //              cudaMemcpyHostToDevice));
  dim3 Dimgridhisto(ceil(imageWidth*imageHeight/512),1,1);
  
  histogram<<<Dimgridhisto , DimBlock>>>(devicegrayImageData, imageWidth*imageHeight, histo );
  //test histogram
   wbCheck(cudaMemcpy(kernelhisto, histo, 256 * sizeof(unsigned int),
                     cudaMemcpyDeviceToHost));
 for(int i=0; i<imageHeight; i++)
    for(int j=0; j<imageWidth; j++){
      int idx = i*imageWidth+j;
      serial_histo[kerneloutput[idx]]++;
  }
  for(int i=0; i<256; i++)
    if(serial_histo[i] == kernelhisto[i])
      printf("%d %d %d\n", i, serial_histo[i], kernelhisto[i]);
  
  //just copy serial_histo to device
 /*  wbCheck(cudaMemcpy(histo, serial_histo, 256* sizeof(unsigned int),
                     cudaMemcpyHostToDevice));  */
  /*
  dim3 DimGridhisto(1,1,1);
  scan<<<DimGridhisto , DimBlock>>>(histo, CDF, imageWidth*imageHeight);
  */
  //do scan in serial
  
  serialcdf[0] = kernelhisto[0]/(imageWidth*imageHeight+0.0f);
  for(int i=1; i<256; i++)
    serialcdf[i] = serialcdf[i-1] + kernelhisto[i]/(imageWidth*imageHeight+0.0f);
  /*wbCheck(cudaMemcpy(kernelequcdf, CDF, 256 * sizeof(unsigned int),
                     cudaMemcpyDeviceToHost));
  faultnum = 0;
  for(int i = 0; i<256;i++){
    //if(kernelequcdf[i]!=serialcdf[i]){
      printf("%d  %f  \n",i,serialcdf[i]);
      //faultnum ++;
    
  }
  */
  wbCheck(cudaMemcpy(CDF, serialcdf, 256* sizeof(float),
                     cudaMemcpyHostToDevice));
  
  
  
  correct<<<DimGrid,DimBlock>>>(CDF, devicecharImageData, imageWidth*imageHeight*imageChannels);
  
  backtofloat<<<DimGrid,DimBlock>>>(devicecharImageData, deviceOutputImageData, imageWidth*imageHeight*imageChannels);
    
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");
  wbTime_start(Copy, "Copying output memory to the CPU");
  wbCheck(cudaMemcpy(hostOutputImageData, deviceOutputImageData, imageWidth*imageHeight*imageChannels * sizeof(float),
                     cudaMemcpyDeviceToHost));
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  cudaFree(deviceInputImageData);
  cudaFree(deviceOutputImageData);
  cudaFree(devicecharImageData);
  cudaFree(devicegrayImageData);
  cudaFree(histo);
  cudaFree(CDF);
  wbTime_stop(GPU, "Freeing GPU Memory");
 
  //above is my code
  
  wbSolution(args, outputImage);

  //@@ insert code here
  free(hostInputImageData);
  free(hostOutputImageData);
  free(test);
  free(kerneloutput);
  free(kernelchar);
  free(kernelhisto);
  return 0;
}
