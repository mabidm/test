

#include "../common/common.h"

void summatrixhost3D(int *A,int *B,int *C, const int nx, const int ny, const int nz){

	    int *ia = A;
	    int *ib = B;
	    int *ic = C;

	 for(int iz=0; iz < nz; iz++){
        //process 2D block 
	    for (int iy = 0; iy < ny; iy++)
	    {
	        for (int ix = 0; ix < nx; ix++)
	        {
	            ic[ix] = ia[ix] + ib[ix];

	        }//ix
	         
            //move pointer to next row
	        ia += nx;
	        ib += nx;
	        ic += nx;
	    }//iy
	    

	 }//iz

	    return;



}
