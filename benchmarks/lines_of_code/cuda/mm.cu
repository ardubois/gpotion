#include <stdio.h>
#include <time.h>
__global__ void gpu_mm(float *a,float *b, float *c, int m, int n, int k)
{ 
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0;
    if( col < k && row < m) 
    {
        for(int i = 0; i < n; i++) 
        {
            sum += a[row * n + i] * b[i * k + col];
        }
        c[row * k + col] = sum;
    }
} 
int main(int argc, char const *argv[])
{   
    int value = atoi(argv[1]);
    int m = value;
    int block_size = 16;
    cudaError_t j_error;
    float *a = (float*) malloc(m*m*sizeof(float));
    float *b = (float*) malloc(m*m*sizeof(float));
    float *c = (float*) malloc(m*m*sizeof(float));
    float *cpu_result = (float*) malloc(m*m*sizeof(float));
    srand(time(0));
   for (int i = 0; i < m; ++i) {
        for (int j = 0; j < m; ++j) {
            a[i * m + j] =  (rand() %(100 -1 + 1)) + 1;
        }
    }
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < m; ++j) {
            b[i * m + j] = (rand() %(100 -1 + 1)) + 1;
        }
    }
    float *d_a, *d_b, *d_c;
    int grid_rows = (m + block_size - 1) / block_size;
    int grid_cols = (m + block_size - 1) / block_size;
    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(block_size, block_size);
   float time;
    cudaEvent_t start, stop;   
     cudaEventCreate(&start) ;
    cudaEventCreate(&stop) ;
    cudaEventRecord(start, 0) ;
    cudaMalloc((void **) &d_a, sizeof(float)*m*m);
     j_error = cudaGetLastError();
    if(j_error != cudaSuccess) printf("Error 1: %s\n", cudaGetErrorString(j_error));
    cudaMalloc((void **) &d_b, sizeof(float)*m*m);
     j_error = cudaGetLastError();
    if(j_error != cudaSuccess) printf("Error 2: %s\n", cudaGetErrorString(j_error));
    cudaMalloc((void **) &d_c, sizeof(float)*m*m);
     j_error = cudaGetLastError();
    if(j_error != cudaSuccess) printf("Error 3: %s\n", cudaGetErrorString(j_error));
    cudaMemcpy(d_a, a, sizeof(float)*m*m, cudaMemcpyHostToDevice);
     j_error = cudaGetLastError();
    if(j_error != cudaSuccess) printf("Error 4: %s\n", cudaGetErrorString(j_error));
    cudaMemcpy(d_b, b, sizeof(float)*m*m, cudaMemcpyHostToDevice);
     j_error = cudaGetLastError();
    if(j_error != cudaSuccess) printf("Error 5: %s\n", cudaGetErrorString(j_error));
    gpu_mm<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, m,m,m);  
    j_error = cudaGetLastError();
    if(j_error != cudaSuccess) printf("Error 6: %s\n", cudaGetErrorString(j_error));
    cudaDeviceSynchronize();
     j_error = cudaGetLastError();
    if(j_error != cudaSuccess) printf("Synchronize: %s\n", cudaGetErrorString(j_error));
    cudaMemcpy(c, d_c, sizeof(float)*m*m, cudaMemcpyDeviceToHost);
    j_error = cudaGetLastError();
    if(j_error != cudaSuccess) printf("Error 7: %s\n", cudaGetErrorString(j_error));
    cudaEventRecord(stop, 0) ;
    cudaEventSynchronize(stop) ;
    cudaEventElapsedTime(&time, start, stop) ;
    printf("cuda\t%d\t%3.1f\n", m,time);
    free(a);
    free(b);
    free(c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}    