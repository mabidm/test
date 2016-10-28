
#include "../common/common.h"

/*nx=ny=width*/

void mulmatricesonhost(int *a,int *b,int *c, const int width){

   int sum;
   int i,j,k;
   for(i=0;i<width;i++)
	   for(j=0;j<width;j++){
		   sum=0;
		   for(k=0;k<width;k++)
			   sum = sum + a[i*width+k] * b[k*width+j];
		   c[i*width+j] = sum;

	   }



}
