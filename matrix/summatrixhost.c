#include "../common/common.h"


/*
 * This example demonstrates a simple matrix addition on host
 */


void summatrixhost(int *A, int *B, int *C, const int nx, const int ny)
{
    int *ia = A;
    int *ib = B;
    int *ic = C;

    for (int iy = 0; iy < ny; iy++)
    {
        for (int ix = 0; ix < nx; ix++)
        {
            ic[ix] = ia[ix] + ib[ix];

        }

        ia += nx;
        ib += nx;
        ic += nx;
    }


}


