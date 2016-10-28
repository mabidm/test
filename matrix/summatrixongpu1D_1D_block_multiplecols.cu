#include "../common/common.h"

/* 
* Each threadblock is mapped to multiple cols
* Each thread is responsible for the whole col
          0  1  0  1 0   1
	 __ __ __ __ __ __ 
        |     |	    |	  |	
        |     |	    |	  |
        |     |	    |	  |
        |     |	    |	  |
        |     |     |	  |
        |     |	    |	  |		
	|__ __|__ __|__ __|
     *
*/

// grid 1D block 1D
__global__ void summatrixongpu1D_1D_block_multiplecols(int *MatA, int *MatB, int *MatC, int nx,
                                 int ny)
{
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;

    if (ix < nx )
        for (int iy = 0; iy < ny; iy++)
        {
            int idx = iy * nx + ix;
            MatC[idx] = MatA[idx] + MatB[idx];
        }


}


