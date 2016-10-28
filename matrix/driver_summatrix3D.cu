

#include "../common/common.h"




#ifdef __cplusplus
extern "C" {
#endif

void summatrixhost3D(int *a,int *b,int *c, const int nx, const int ny, const int nz);



#ifdef __cplusplus
}
#endif


__global__ void summatrixongpu2D_3D(int *MatA, int *MatB, int *MatC, int nx, int ny, int nz);
__global__ void summatrixongpu3D_3D(int *MatA, int *MatB, int *MatC, int nx, int ny, int nz);







void driver_summatrix3D(int argc, char **argv) {

	printf("%s starting...\n",argv[0]);

	//default size

	unsigned int nx = 1 << 4;
	unsigned int ny = 1 << 4;
	unsigned int nz = 1 << 2;



	if (argc > 3){ //user given for THREEDIM matrix
		nx = atoi(argv[1]);
		ny = atoi(argv[2]);
		nz = atoi(argv[3]);
		printf("Matrix Addition Size: %d X %d X %d\n",nx, ny, nz);
	}


    // set up device
	int dev = 0;
	cudaDeviceProp deviceProp;
	CHECK(cudaGetDeviceProperties(&deviceProp, dev));
	printf("Using Device %d: %s\n", dev, deviceProp.name);
	CHECK(cudaSetDevice(dev));

    //total bytes for each matrix
	int nxyz = nx * ny * nz;
	size_t nbytes = nxyz * sizeof(int);

	//host
	int *h_a;
	int *h_b;
	int *h_c;
	int *h_gpures;

	h_a = (int*) malloc(nbytes);SAFE_MALLOC_CALL(h_a);
	h_b = (int*) malloc(nbytes);SAFE_MALLOC_CALL(h_b);
	h_c = (int*) malloc(nbytes);SAFE_MALLOC_CALL(h_c);
	h_gpures = (int*) malloc(nbytes); SAFE_MALLOC_CALL(h_gpures);



	initialdata(h_a, nxyz);
	initialdata(h_b, nxyz);
	memset(h_gpures, 0, nbytes);



//GPU
	int *d_a;
	int *d_b;
	int *d_c;

	CHECK(cudaMalloc((void**) &d_a, nbytes));
	CHECK(cudaMalloc((void**) &d_b, nbytes));
	CHECK(cudaMalloc((void**) &d_c, nbytes));

	CHECK(cudaMemcpy(d_a, h_a, nbytes, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_b, h_b, nbytes, cudaMemcpyHostToDevice));
	//CHECK(cudaMemset( d_c, 0, nbytes ));

	dim3 block(8, 8, 4);

	dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

         double start, elapsed_time;

	//host
	start = cpuseconds();
	summatrixhost3D(h_a, h_b, h_c, nx, ny, nz);
	elapsed_time = cpuseconds() - start;
	printf("CPU Time: %-10f\n", elapsed_time);
	
        
	//kernel summatrixongpu2D_3D launching
	start = cpuseconds();
	summatrixongpu2D_3D<<<grid, block>>>(d_a, d_b, d_c, nx, ny, nz);
	CHECK(cudaGetLastError());//ERROR in the kernel launch
	CHECK(cudaDeviceSynchronize());
	elapsed_time = cpuseconds() - start;
	printf("summatrixongpu2D_3D <<< (%d,%d,%d), block(%d,%d,%d)>>>:  %-10f\n",
			 grid.x, grid.y, grid.z,block.x, block.y, block.z, elapsed_time);

	CHECK(cudaMemcpy(h_gpures, d_c, nbytes, cudaMemcpyDeviceToHost));
        checkresint(h_c, h_gpures, nxyz);

        //3D grid
         //threadblock
         block.x = 8;
         block.x = 8;  
         block.x = 2;  
         //grid  
       	 grid.x = (nx + block.x - 1) / block.x;
         grid.y = (ny + block.y - 1) / block.y;
         grid.z = (nz + block.z - 1) / block.z;

         //kernel summatrixongpu3D_3D launching
	start = cpuseconds();
	summatrixongpu3D_3D<<<grid, block>>>(d_a, d_b, d_c, nx, ny, nz);
	CHECK(cudaGetLastError());//ERROR in the kernel launch
	CHECK(cudaDeviceSynchronize());
	elapsed_time = cpuseconds() - start;
	printf("summatrixongpu3D_3D <<< (%d,%d,%d), block(%d,%d,%d)>>>:  %-10f\n",
			 grid.x, grid.y, grid.z,block.x, block.y, block.z, elapsed_time);
	CHECK(cudaMemcpy(h_gpures, d_c, nbytes, cudaMemcpyDeviceToHost));
        checkresint(h_c, h_gpures, nxyz);



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



