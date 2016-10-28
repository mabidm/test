#include "../common/common.h"

/* 
* Each threadblock is mapped to a single row
* Each thread is responsible for a single element of a row
	 _______________________
Block 0	|_______________________|
     1	|_______________________|	
     2	|_______________________|
     3	|_______________________|
     4	|_______________________|
     5	|_______________________|
*
*/

// grid 1D block 1D
__global__ void summatrixongpu1D_1D_block_row(int *MatA, int *MatB, int *MatC, int nx,
                                 int ny)
{
    unsigned int ix = threadIdx.x;
    unsigned int iy = blockIdx.y;
    unsigned int idx = iy * nx + ix;

    if (ix < nx && iy < ny)
        MatC[idx] = MatA[idx] + MatB[idx];


}


