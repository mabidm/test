#include "../common/common.h"

/*
* Prints the contents of a matrix
*
*/
void printmatrix(int *C, const int nx, const int ny)
{
    int *ic = C;

    for (int iy = 0; iy < ny; iy++)
    {
        for (int ix = 0; ix < nx; ix++)
        {
            printf("%d ", ic[ix]);

        }

        ic += nx;
        printf("\n");
    }


}


