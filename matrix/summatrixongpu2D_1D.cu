#include "../common/common.h"

// grid 2D block 1D
__global__ void summatrixongpu2D_1D(int *MatA, int *MatB, int *MatC, int nx,
                                  int ny)
{
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = iy * nx + ix;

    if (ix < nx && iy < ny)
        MatC[idx] = MatA[idx] + MatB[idx];
}



