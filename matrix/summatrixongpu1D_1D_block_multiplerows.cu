#include "../common/common.h"

/* 
* Each threadblock is mapped to multiple rows
* Each thread is responsible for the whole row
	 _______________________
     0	|  			|
     1	|_______________________|	
     0	|   			|
     1	|_______________________|
     0	|			|
     1	|_______________________|
*
*/

// grid 1D block 1D
__global__ void summatrixongpu1D_1D_block_multiplerows(int *MatA, int *MatB, int *MatC, int nx,
                                 int ny)
{
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;

    if (iy < ny )
        for (int ix = 0; ix < nx; ix++)
        {
            int idx = iy * nx + ix;
            MatC[idx] = MatA[idx] + MatB[idx];
        }


}


