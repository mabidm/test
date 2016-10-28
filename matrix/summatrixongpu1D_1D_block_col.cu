#include "../common/common.h"

/* 
* Each threadblock is mapped to a single col
* Each thread is responsible for a single element of a col
 Block   0  1  2  3  4   5
	 __ __ __ __ __ __ 
        |  |  |	 |  |  |  |	
        |  |  |	 |  |  |  |
        |  |  |	 |  |  |  |
        |  |  |	 |  |  |  |
        |  |  |  |  |  |  |
        |  |  |	 |  |  |  |		
	|__|__|__|__|__|__|
     *
*/

// grid 1D block 1D
__global__ void summatrixongpu1D_1D_block_col(int *MatA, int *MatB, int *MatC, int nx,
                                 int ny)
{
    unsigned int ix =  blockIdx.x;
    unsigned int iy =  threadIdx.y;
    unsigned int idx =  iy * nx + ix;


     if (ix < nx && iy < ny)
        MatC[idx] = MatA[idx] + MatB[idx];


}


