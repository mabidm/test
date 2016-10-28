
#include "../common/common.h"

__global__ void summatrixongpu3D_3D(int *a,int *b,int *c, const int nx, const int ny, const int nz ){

   //mapping threadIdx and blockInd to 3D data index
   int ix = blockIdx.x * blockDim.x + threadIdx.x;
   int iy = blockIdx.y * blockDim.y + threadIdx.y;
   int iz = blockIdx.z * blockDim.z + threadIdx.z;

   //3D data index to 1D data index
   int index = iz * nx * ny + iy * nx +ix;

   if(ix < nx  && iy < ny && iz < nz){
	   c[index] = a[index] + b[index];


  }

}
