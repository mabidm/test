#include "../common/common.h"

#ifdef __cplusplus
extern "C" {
#endif


void summatrixhost(int *A, int *B, int *C, const int nx, const int ny);
//void summatrixhost3D(int *A, int *B, int *C, const int nx, const int ny, const int nz);




#ifdef __cplusplus
}
#endif

__global__ void summatrixongpu2D_2D(int *MatA, int *MatB, int *MatC, int nx,  int ny);
__global__ void summatrixongpu2D_1D(int *MatA, int *MatB, int *MatC, int nx,  int ny);


__global__ void summatrixongpu1D_1D_block_row(int *MatA, int *MatB, int *MatC, int nx, int ny);
__global__ void summatrixongpu1D_1D_block_col(int *MatA, int *MatB, int *MatC, int nx, int ny);
__global__ void summatrixongpu1D_1D_block_multiplerows(int *MatA, int *MatB, int *MatC, int nx,
                                 int ny);
__global__ void summatrixongpu1D_1D_block_multiplecols(int *MatA, int *MatB, int *MatC, int nx,
                                 int ny);




void driver_summatrix(int argc, char **argv)
{
    printf("%s Starting...\n", argv[0]);

    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    // set up size of matrix
    int nx = 1 << 10;
    int ny = 1 << 10;


    //user defined size
    if (argc > 2) {
         nx = atoi(argv[1]);
	 ny = atoi(argv[2]);

    }

    //total bytes
    int nxy = nx * ny;
    int nBytes = nxy * sizeof(int);
    printf("Matrix size: nx %d ny %d \n", nx, ny);

    // malloc host memory
    int *h_A, *h_B, *hostRef, *gpuRef;
    h_A = (int *)malloc(nBytes);SAFE_MALLOC_CALL(h_A);
    h_B = (int *)malloc(nBytes); SAFE_MALLOC_CALL(h_B);
    hostRef = (int *)malloc(nBytes); SAFE_MALLOC_CALL(hostRef);
    gpuRef = (int *)malloc(nBytes); SAFE_MALLOC_CALL(gpuRef);
    
// initialize data at host side

    initialdata(h_A,nxy);
    initialdata(h_B,nxy);
    //reset output to 0
    memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes);


    //device memory allocation  

    int *d_MatA, *d_MatB, *d_MatC;
    CHECK(cudaMalloc((int **)&d_MatA, nBytes));
    CHECK(cudaMalloc((int **)&d_MatB, nBytes));
    CHECK(cudaMalloc((int **)&d_MatC, nBytes));
    

    //copy host --> device

    CHECK(cudaMemcpy(d_MatA, h_A, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_MatB, h_B, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemset(d_MatC,0, nBytes));

    //block dimension
    int dimx = 16;
    int dimy = 16;

    dim3 block(dimx, dimy);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);
     

    double start, elapsed_time;

     //cpu matrix sum 2D
    start = cpuseconds();
    summatrixhost(h_A, h_B, hostRef, nx, ny);
    elapsed_time = cpuseconds() - start;
    printf("summatrixhost elapsed %f sec\n", elapsed_time );

    //kernel launch summatrixongpu2D_2D
   start  = cpuseconds(); 
   summatrixongpu2D_2D<<<grid,block>>>(d_MatA,d_MatB,d_MatC,nx,ny);
   CHECK(cudaGetLastError());
   CHECK(cudaDeviceSynchronize());
   elapsed_time = cpuseconds() - start ;
   printf("summatrixongpu2D_2D<<<(%d,%d),(%d,%d)>>> , elapsed: %fsec\n",  grid.x,grid.y,block.x,block.y,elapsed_time);
   //copy device to host
   CHECK(cudaMemcpy(gpuRef, d_MatC, nBytes, cudaMemcpyDeviceToHost));
   //compare host and device result
    checkresint(hostRef, gpuRef, nxy);

    ////kernel launch summatrixongpu2D_1D
    block.x = dimx;
    block.y = 1;
    block.z = 1;
    grid.x = (nx + block.x -1) / block.x;
    grid.y = ny;
    grid.z = 1;
   start  = cpuseconds(); 
   summatrixongpu2D_1D<<<grid,block>>>(d_MatA,d_MatB,d_MatC,nx,ny);
   CHECK(cudaGetLastError());
   CHECK(cudaDeviceSynchronize());
   elapsed_time = cpuseconds() - start ;
   printf("summatrixongpu2D_1D<<<(%d,%d),(%d,%d)>>> , elapsed: %fsec\n",  grid.x,grid.y,block.x,block.y,elapsed_time);
   //copy device to host
   CHECK(cudaMemcpy(gpuRef, d_MatC, nBytes, cudaMemcpyDeviceToHost));
   //compare host and device result
    checkresint(hostRef, gpuRef, nxy);

     //summatrixongpu1D_1D_block_col
    block.x = 1;
    block.y = ny;
    block.z = 1;
    grid.x = nx;
    grid.y = 1;
    grid.z = 1;

    start = cpuseconds();
    summatrixongpu1D_1D_block_col<<<grid, block>>>(d_MatA, d_MatB, d_MatC, nx, ny);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    elapsed_time = cpuseconds() - start;
    printf("summatrixongpu1D_1D_block_col <<<(%d,%d), (%d,%d)>>>, elapsed %f sec\n",
           grid.x, grid.y, block.x, block.y, elapsed_time);
     //copy device to host
   CHECK(cudaMemcpy(gpuRef, d_MatC, nBytes, cudaMemcpyDeviceToHost));
   //compare host and device result
    checkresint(hostRef, gpuRef, nxy);


    //summatrixongpu1D_1D_block_multiplecols
    block.x = 4;
    block.y = 1;
    block.z = 1;
    grid.x = (nx + block.x -1) / block.x;
    grid.y = 1;
    grid.z = 1;

    start = cpuseconds();
    summatrixongpu1D_1D_block_multiplecols<<<grid, block>>>(d_MatA, d_MatB, d_MatC, nx, ny);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    elapsed_time = cpuseconds() - start;
    printf("summatrixongpu1D_1D_block_multiplecols <<< (%d,%d),(%d,%d)>>>, elapsed %f sec\n",
           grid.x, grid.y, block.x, block.y, elapsed_time);
     //copy device to host
   CHECK(cudaMemcpy(gpuRef, d_MatC, nBytes, cudaMemcpyDeviceToHost));
   //compare host and device result
    checkresint(hostRef, gpuRef, nxy);
	


   //summatrixongpu1D_1D_block_row
    block.x = nx;
    block.y = 1;
    block.z = 1;
    grid.x = 1;
    grid.y = ny;
    grid.z = 1;

    start = cpuseconds();
    summatrixongpu1D_1D_block_row<<<grid, block>>>(d_MatA, d_MatB, d_MatC, nx, ny);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    elapsed_time = cpuseconds() - start;
    printf("summatrixongpu1D_1D_block_row <<<(%d,%d), (%d,%d)>>>,elapsed %f sec\n",
           grid.x, grid.y, block.x, block.y, elapsed_time);
     //copy device to host
   CHECK(cudaMemcpy(gpuRef, d_MatC, nBytes, cudaMemcpyDeviceToHost));
   //compare host and device result
    checkresint(hostRef, gpuRef, nxy);
	
    
 //summatrixongpu1D_1D_block_multiplerows
    block.x = 1;
    block.y = 4;
    block.z = 1;
    grid.x = 1;
    grid.y = (ny + block.y -1) / block.y;
    grid.z = 1;

    start = cpuseconds();
    summatrixongpu1D_1D_block_multiplerows<<<grid, block>>>(d_MatA, d_MatB, d_MatC, nx, ny);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    elapsed_time = cpuseconds() - start;
    printf("summatrixongpu1D_1D_block_multiplerows <<<(%d,%d), (%d,%d)>>>,elapsed %f sec\n",
           grid.x, grid.y, block.x, block.y, elapsed_time);
     //copy device to host
   CHECK(cudaMemcpy(gpuRef, d_MatC, nBytes, cudaMemcpyDeviceToHost));
   //compare host and device result
    checkresint(hostRef, gpuRef, nxy);





    
   //free host memory
    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);
    //free device memory
    CHECK(cudaFree(d_MatA));
    CHECK(cudaFree(d_MatB));
    CHECK(cudaFree(d_MatC));


    CHECK(cudaDeviceReset());
}
