
#include "../common/common.h"

/*nx=ny=width*/

__global__ void mulmatricesongpu(int *a,int *b,int *c, const int width){

   int sum=0;
   int k;
   int col = blockIdx.x * blockDim.x + threadIdx.x;//ix
   int row = blockIdx.y * blockDim.y + threadIdx.y;//iy

   if(row < width && col < width) {

	 for(k=0;k<width;k++)
	   sum = sum + a[row*width+k] * b[k*width+col];
	 c[row*width+col] = sum;

   }



}
