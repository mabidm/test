
#include "../common/common.h"





#ifdef __cplusplus
extern "C" {
#endif


void mulmatricesonhost(int *a,int *b,int *c, const int width);

#ifdef __cplusplus
}
#endif

__global__ void mulmatricesongpu(int *a,int *b,int *c, const int width);
__global__ void mulmatricesongpu_shared(int *a,int *b,int *c, const int width);
__global__ void mulmatricesongpu_shared_pad(int *a,int *b,int *c, const int width);


void driver_mulmatrices(int argc, char **argv) {

        //read args from command line
	printf("%s starting...\n",argv[0]);
	int width = 1 << 10;
        if(argc > 1) width = atoi(argv[1]);
        printf("Matrix Multiplication Size: %d X %d\n",width,width);

	// set up device
	int dev = 0;
	cudaDeviceProp deviceProp;
	CHECK(cudaGetDeviceProperties(&deviceProp, dev));
	printf("Using Device %d: %s\n", dev, deviceProp.name);
	CHECK(cudaSetDevice(dev));


	size_t nbytes = width * width * sizeof(int);

	//allocate host memory
	int *h_a;
	int *h_b;
	int *h_c;
	int *h_gpures;
	
    h_a = (int*) malloc(nbytes);
	SAFE_MALLOC_CALL(h_a);
        
	h_b = (int*) malloc(nbytes);
	SAFE_MALLOC_CALL(h_b);
    
    h_c = (int*) malloc(nbytes);
	SAFE_MALLOC_CALL(h_c);
    
    h_gpures = (int*) malloc(nbytes);
	SAFE_MALLOC_CALL(h_gpures);
    
	initialdata(h_a, width  * width );
	initialdata(h_b, width  * width);
	memset(h_gpures, 0, nbytes);

/*
	int i;
	 for (i=0; i < width*width; i++){
	 printf("%f ",h_a[i]);
	 printf("%f \n",h_b[i]);
	 }
*/

	//allocate device memory
	int *d_a;
	int *d_b;
	int *d_c;

	CHECK(cudaMalloc((void**) &d_a, nbytes));
	CHECK(cudaMalloc((void**) &d_b, nbytes));
	CHECK(cudaMalloc((void**) &d_c, nbytes));

	CHECK(cudaMemcpy(d_a, h_a, nbytes, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_b, h_b, nbytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemset(d_c, 0, nbytes));



	dim3 block(TILE_WIDTH_MULTIPLICATION, TILE_WIDTH_MULTIPLICATION, 1);
	dim3 grid((width + block.x - 1) / block.x, (width + block.y - 1) / block.y, 1);

        //based on global memory 
	double start = cpuseconds();
	mulmatricesongpu<<<grid, block>>>(d_a, d_b, d_c, width);
	CHECK(cudaDeviceSynchronize());
	double t = cpuseconds() - start;
	printf(	"mulmatricesongpu time: %f\n",	t);

        //based on smem 
	start = cpuseconds();
	mulmatricesongpu_shared<<<grid, block>>>(d_a, d_b, d_c, width);
	CHECK(cudaDeviceSynchronize());
	t = cpuseconds() - start;
	printf(	"mulmatricesongpu_shared time: %f\n",	t);

        //based on smem with padding
	start = cpuseconds();
	mulmatricesongpu_shared_pad<<<grid, block>>>(d_a, d_b, d_c, width);
	CHECK(cudaDeviceSynchronize());
	t = cpuseconds() - start;
	printf(	"mulmatricesongpu_shared_pad time: %f\n",	t);

	
	CHECK(cudaMemcpy(h_gpures, d_c, nbytes, cudaMemcpyDeviceToHost));


 	start = cpuseconds();
	mulmatricesonhost(h_a, h_b, h_c, width);
	t = cpuseconds() - start;
	printf("mulmatricesoncpu Time: %f\n", t);




	checkresint(h_c, h_gpures, width  * width);



        //free host memory
	free(h_a);
	free(h_b);
	free(h_c);
	free(h_gpures);

	//free device memory
	CHECK(cudaFree(d_a));
	CHECK(cudaFree(d_b));
	CHECK(cudaFree(d_c));

	CHECK(cudaDeviceReset());


}


