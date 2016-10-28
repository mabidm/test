#include "../common/common.h"

// grid 2D block 3D
__global__ void summatrixongpu2D_3D(int *MatA, int *MatB, int *MatC, int nx, int ny, int nz)
{
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int iz = threadIdx.z ;
    
    unsigned int idx = iz*nx*ny + iy * nx + ix;

    if (ix < nx && iy < ny && iz < nz)
        MatC[idx] = MatA[idx] + MatB[idx];
        
}


