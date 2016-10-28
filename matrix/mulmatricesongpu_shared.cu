
#include "../common/common.h"
#define IPAD  2

/*nx=ny=width*/

__global__ void mulmatricesongpu_shared(int *a,int *b,int *c, const int width){


	__shared__ int as[TILE_WIDTH_MULTIPLICATION][TILE_WIDTH_MULTIPLICATION];
	__shared__ int bs[TILE_WIDTH_MULTIPLICATION][TILE_WIDTH_MULTIPLICATION];

   int sum=0;
   int k;
   
   //cal data index
   int col = blockIdx.x * TILE_WIDTH_MULTIPLICATION + threadIdx.x;
   int row = blockIdx.y * TILE_WIDTH_MULTIPLICATION + threadIdx.y;
   
   int subblock;
   
   //index for shared memory
   int tx = threadIdx.x;
   int ty = threadIdx.y;
   
   if(row < width && col < width) { //assume all matrices have same dimension
	   for (subblock =0; subblock < width/TILE_WIDTH_MULTIPLICATION; subblock++){
		   as[ty][tx] = a[ row*width + (subblock*TILE_WIDTH_MULTIPLICATION+tx)]; // col = (subblock*TILE_WIDTH_MULTIPLICATION+tx)
		   bs[ty][tx] = b[ (subblock*TILE_WIDTH_MULTIPLICATION+ty)*width + col];// row = (subblock*TILE_WIDTH_MULTIPLICATION+ty)
		   //printf("B(%d,%d),T(%d,%d), a = %d, b = %d\n",blockIdx.x,blockIdx.y,threadIdx.x,threadIdx.y,
			//	   a[ row*width + (subblock*TILE_WIDTH_MULTIPLICATION+tx)], b[ (subblock*TILE_WIDTH_MULTIPLICATION+ty)*width + col]);
		   __syncthreads();
		   //for(int i=0;i<TILE_WIDTH_MULTIPLICATION;i++)
		   //  for(int j=0;j<TILE_WIDTH_MULTIPLICATION;j++)
			//	   printf("as[%d][%d]= %f, bs[%d][%d]=%f\n",i,j,as[i][j],i,j,bs[i][j]);

		   for(k=0;k<TILE_WIDTH_MULTIPLICATION;k++)
			   sum = sum + as[ty][k] * bs[k][tx];
			   //wait until all threads are done with this block
		   __syncthreads();
		}
	   c[row*width+col] = sum;
	   //if(blockIdx.x==127)
	   //printf("B(%d,%d),T(%d,%d) ",blockIdx.x,blockIdx.y,threadIdx.x,threadIdx.y);

   }//if




}

///with padding

__global__ void mulmatricesongpu_shared_pad(int *a,int *b,int *c, const int width){


	__shared__ int as[TILE_WIDTH_MULTIPLICATION][TILE_WIDTH_MULTIPLICATION+IPAD];
	__shared__ int bs[TILE_WIDTH_MULTIPLICATION][TILE_WIDTH_MULTIPLICATION+IPAD];

   int sum=0;
   int k;
   
   //cal data index
   int col = blockIdx.x * TILE_WIDTH_MULTIPLICATION + threadIdx.x;
   int row = blockIdx.y * TILE_WIDTH_MULTIPLICATION + threadIdx.y;
   
   int subblock;
   
   //index for shared memory
   int tx = threadIdx.x;
   int ty = threadIdx.y;
   
   if(row < width && col < width) { //assume all matrices have same dimension
	   for (subblock =0; subblock < width/TILE_WIDTH_MULTIPLICATION; subblock++){
		   as[ty][tx] = a[ row*width + (subblock*TILE_WIDTH_MULTIPLICATION+tx)]; // col = (subblock*TILE_WIDTH_MULTIPLICATION+tx)
		   bs[ty][tx] = b[ (subblock*TILE_WIDTH_MULTIPLICATION+ty)*width + col];// row = (subblock*TILE_WIDTH_MULTIPLICATION+ty)
		   //printf("B(%d,%d),T(%d,%d), a = %d, b = %d\n",blockIdx.x,blockIdx.y,threadIdx.x,threadIdx.y,
			//	   a[ row*width + (subblock*TILE_WIDTH_MULTIPLICATION+tx)], b[ (subblock*TILE_WIDTH_MULTIPLICATION+ty)*width + col]);
		   __syncthreads();
		   //for(int i=0;i<TILE_WIDTH_MULTIPLICATION;i++)
		   //  for(int j=0;j<TILE_WIDTH_MULTIPLICATION;j++)
			//	   printf("as[%d][%d]= %f, bs[%d][%d]=%f\n",i,j,as[i][j],i,j,bs[i][j]);

		   for(k=0;k<TILE_WIDTH_MULTIPLICATION;k++)
			   sum = sum + as[ty][k] * bs[k][tx];
			   //wait until all threads are done with this block
		   __syncthreads();
		}
	   c[row*width+col] = sum;
	   //if(blockIdx.x==127)
	   //printf("B(%d,%d),T(%d,%d) ",blockIdx.x,blockIdx.y,threadIdx.x,threadIdx.y);

   }//if




}


/*

global__ void MatMul(int* A, int* B, int* C, int ARows, int ACols, int BRows, int BCols, int CRows, int CCols) {

  int CValue = 0;
  int Row = blockIdx.y*TILE_DIM + threadIdx.y;
  int Col = blockIdx.x*TILE_DIM + threadIdx.x;

  __shared__ int As[TILE_DIM][TILE_DIM];
  __shared__ int Bs[TILE_DIM][TILE_DIM];

    for (int k = 0; k < (TILE_DIM + ACols - 1)/TILE_DIM; k++) {

      if (k*TILE_DIM + threadIdx.x < ACols && Row < ARows)
    	  As[threadIdx.y][threadIdx.x] = A[Row*ACols + k*TILE_DIM + threadIdx.x];
      else As[threadIdx.y][threadIdx.x] = 0.0;

      if (k*TILE_DIM + threadIdx.y < BRows && Col < BCols)
    	  Bs[threadIdx.y][threadIdx.x] = B[(k*TILE_DIM + threadIdx.y)*BCols + Col];
      else Bs[threadIdx.y][threadIdx.x] = 0.0;

      __syncthreads();

      for (int n = 0; n < TILE_DIM; ++n) CValue += As[threadIdx.y][n] * Bs[n][threadIdx.x];

      __syncthreads();

  }

  if (Row < CRows && Col < CCols)
  C[((blockIdx.y * blockDim.y + threadIdx.y)*CCols)+(blockIdx.x*blockDim.x)+threadIdx.x]=CValue;

}
*/
