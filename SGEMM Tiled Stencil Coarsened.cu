#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

// Compute C = A * B
__global__ void matrixMultiply_kernel(float *A, float *B, float *C,
                                      int numARows, int numAColumns,
                                      int numBRows, int numBColumns,
                                      int numCRows, int numCColumns) {
  //@@ Insert code to implement matrix multiplication here
  //@@ You have to perform register tiling for this MP
	
	#define outputC(i,j) C[i*numCColumns+j]
	const int TILE_WIDTH = 12;	
	
	float sumValue = 0;
	int iterator,iteratorB;
	float first,second, third;
	/*
	*	Thread Coarning LEvel
	*/
	int CoarsedLevel = 4;                   
	int RowsToLoad = TILE_WIDTH/CoarsedLevel;       
	
	
	__shared__ float B_Y[TILE_WIDTH];
	int Row = blockIdx.x * blockDim.x + threadIdx.x;
	
	
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	
	for(iterator=0;iterator<numAColumns;iterator+=RowsToLoad)
	{
			
		if(Row<numARows){
			first = A[iterator*numCRows+Row];
			second = A[(iterator+1)*numCRows+Row];
        third = A[(iterator+2)*numCRows+Row];
		}
		else
		{
			first =0;
			second=0;
      third = 0;
		}
			
			__syncthreads();
		
		for(iteratorB=0;iteratorB<numCColumns;iteratorB+=CoarsedLevel)
		{
			/*
			*	Here, we first read elements, based on value. For each thread, we read one value, depending on its thread Index.
			*/
			if(threadIdx.x<CoarsedLevel && iteratorB<numCColumns &&(iterator)<numAColumns)
			{
					
					B_Y[tx] = B[tx+ iteratorB + iterator*numCColumns];
					
			}
			else if(CoarsedLevel<=threadIdx.x && threadIdx.x<2*CoarsedLevel && threadIdx.x<TILE_WIDTH && iteratorB<numCColumns && (iterator+1)<numAColumns)
			{
					B_Y[tx] = B[(tx-CoarsedLevel)+ iteratorB + numCColumns+(iterator*numCColumns)];
					
			}
      else if(CoarsedLevel*2<=threadIdx.x && threadIdx.x<TILE_WIDTH && iteratorB<numCColumns && (iterator+2)<numAColumns)
			{
					B_Y[tx] = B[(tx-CoarsedLevel*2)+ iteratorB + numCColumns*2+(iterator*numCColumns)];
					
			}
			else
			{
					B_Y[tx] = 0;	
			}
			
			
			__syncthreads();
			/*
			*	This loads elements in Shared Tile matrix.
			*/
			
			
			/*
			*	Loaded All elements and synced threads, now we Calculate 4 elements.
			*/
			for(int fTile=0;fTile<CoarsedLevel;fTile++)
			{
				
				atomicAdd(&C[Row*numCColumns+fTile+iteratorB], first*B_Y[fTile]);
				atomicAdd(&C[Row*numCColumns+fTile+iteratorB], second * B_Y[fTile+CoarsedLevel]);
          atomicAdd(&C[Row*numCColumns+fTile+iteratorB], third * B_Y[fTile+2*CoarsedLevel]);
				
			}
			
			__syncthreads();
		}
		
		__syncthreads();
	}
	
	
}

static void matrixMultiply(float *A, float *B, float *C, int numARows,
                           int numAColumns, int numBRows, int numBColumns,
                           int numCRows, int numCColumns) {
  //@@ Insert code to launch matrix multiplication
	
	dim3 DimGrid(ceil((numCRows-1)/12.0),1,1);
	dim3 DimBlock(12,1,1);
		/*
		*calling Matrix Multiply Kernel This is a critical part. Taking Tile_size as 8. 
		*/
	matrixMultiply_kernel<<<DimGrid,DimBlock>>>(A,B,C,numARows,numAColumns, numBRows, numBColumns,
                           numCRows, numCColumns);
	
}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostA; // The A matrix
  float *hostB; // The B matrix
  float *hostC; // The output C matrix
  float *deviceA;
  float *deviceB;
  float *deviceC;
  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;    // number of rows in the matrix C (you have to set this)
  int numCColumns; // number of columns in the matrix C (you have to set
                   // this)

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostA = (float *)wbImport(wbArg_getInputFile(args, 0), &numAColumns,
                            &numARows);
  hostB = (float *)wbImport(wbArg_getInputFile(args, 1), &numBRows,
                            &numBColumns);
  numCRows = numARows;
  numCColumns = numBColumns;
  hostC = (float *)malloc(sizeof(float) * numCRows * numCColumns);
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
  wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);

  wbTime_start(GPU, "Allocating GPU memory.");
  cudaMalloc((void **)&deviceA, sizeof(float) * numARows * numAColumns);
  cudaMalloc((void **)&deviceB, sizeof(float) * numBRows * numBColumns);
  cudaMalloc((void **)&deviceC, sizeof(float) * numCRows * numCColumns);
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  cudaMemcpy(deviceA, hostA, sizeof(float) * numARows * numAColumns,
             cudaMemcpyHostToDevice);
  cudaMemcpy(deviceB, hostB, sizeof(float) * numBRows * numBColumns,
             cudaMemcpyHostToDevice);
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  wbTime_start(Compute, "Performing CUDA computation");
  matrixMultiply(deviceA, deviceB, deviceC, numARows, numAColumns,
                 numBRows, numBColumns, numCRows, numCColumns);
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  cudaMemcpy(hostC, deviceC, sizeof(float) * numCRows * numCColumns,
             cudaMemcpyDeviceToHost);
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  cudaFree(deviceA);
  cudaFree(deviceB);
  cudaFree(deviceC);
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostC, numCRows, numCColumns);

  free(hostA);
  free(hostB);
  free(hostC);

  return 0;
}
