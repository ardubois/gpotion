#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "cuda.h"
#include <string.h>
#include <math.h>
#define MAXBLOCKSIZE 512
#define BLOCK_SIZE_XY 4
int Size;
float *a, *b, *finalVec;
float *m;
// create both matrix and right hand side, Ke Wang 2013/08/12 11:51:06
void
create_matrix(float *m, int size){
  int i,j;
  float lamda = -0.01;
  float coe[2*size-1];
  float coe_i =0.0;
  for (i=0; i < size; i++)
    {
      coe_i = 10*exp(lamda*i); 
      j=size-1+i;     
      coe[j]=coe_i;
      j=size-1-i;     
      coe[j]=coe_i;
    }
  for (i=0; i < size; i++) {
      for (j=0; j < size; j++) {
	m[i*size+j]=coe[size-1-i+j];
      }
  }
}
int main(int argc, char *argv[])
{
          Size = atoi(argv[1]);
	      a = (float *) malloc(Size * Size * sizeof(float));
	      create_matrix(a, Size);
	      b = (float *) malloc(Size * sizeof(float));
	      for (int j =0; j< Size; j++)
	    	b[j]=1.0;
	      m = (float *) malloc(Size * Size * sizeof(float));
    InitPerRun();
	float time;
    cudaEvent_t start, stop;   
     cudaEventCreate(&start) ;
    cudaEventCreate(&stop) ;
    cudaEventRecord(start, 0) ;
    ForwardSub();
    cudaEventRecord(stop, 0) ;
    cudaEventSynchronize(stop) ;
    cudaEventElapsedTime(&time, start, stop) ;
    printf("CUDA\t%d\t%3.1f\n", Size,time);
    BackSub();
   free(m);
    free(a);
    free(b);
}
void InitPerRun() 
{
	int i;
	for (i=0; i<Size*Size; i++)
			*(m+i) = 0.0;
}
__global__ void Fan1(float *m_cuda, float *a_cuda, int Size, int t)
{   
	if(threadIdx.x + blockIdx.x * blockDim.x >= Size-1-t) return;
	*(m_cuda+Size*(blockDim.x*blockIdx.x+threadIdx.x+t+1)+t) = *(a_cuda+Size*(blockDim.x*blockIdx.x+threadIdx.x+t+1)+t) / *(a_cuda+Size*t+t);
}
__global__ void Fan2(float *m_cuda, float *a_cuda, float *b_cuda,int Size, int j1, int t)
{
	if(threadIdx.x + blockIdx.x * blockDim.x >= Size-1-t) return;
	if(threadIdx.y + blockIdx.y * blockDim.y >= Size-t) return;
    int xidx = blockIdx.x * blockDim.x + threadIdx.x;
	int yidx = blockIdx.y * blockDim.y + threadIdx.y;
		a_cuda[Size*(xidx+1+t)+(yidx+t)] -= m_cuda[Size*(xidx+1+t)+t] * a_cuda[Size*t+(yidx+t)];
	if(yidx == 0){
		b_cuda[xidx+1+t] -= m_cuda[Size*(xidx+1+t)+(yidx+t)] * b_cuda[t];
	}
}
void ForwardSub()
{
	int t;
    float *m_cuda,*a_cuda,*b_cuda;
	cudaMalloc((void **) &m_cuda, Size * Size * sizeof(float));
	cudaMalloc((void **) &a_cuda, Size * Size * sizeof(float));
	cudaMalloc((void **) &b_cuda, Size * sizeof(float));	
	cudaMemcpy(m_cuda, m, Size * Size * sizeof(float),cudaMemcpyHostToDevice );
	cudaMemcpy(a_cuda, a, Size * Size * sizeof(float),cudaMemcpyHostToDevice );
	cudaMemcpy(b_cuda, b, Size * sizeof(float),cudaMemcpyHostToDevice );
	int block_size,grid_size;
	block_size = MAXBLOCKSIZE;
	grid_size = (Size/block_size) + (!(Size%block_size)? 0:1);
	dim3 dimBlock(block_size);
	dim3 dimGrid(grid_size);
	int blockSize2d, gridSize2d;
	blockSize2d = BLOCK_SIZE_XY;
	gridSize2d = (Size/blockSize2d) + (!(Size%blockSize2d?0:1)); 
	dim3 dimBlockXY(blockSize2d,blockSize2d);
	dim3 dimGridXY(gridSize2d,gridSize2d);
	for (t=0; t<(Size-1); t++) {
		Fan1<<<dimGrid,dimBlock>>>(m_cuda,a_cuda,Size,t);
		cudaDeviceSynchronize();
		Fan2<<<dimGridXY,dimBlockXY>>>(m_cuda,a_cuda,b_cuda,Size,Size-t,t);
		cudaDeviceSynchronize();
		checkCUDAError("Fan2");
	}
	cudaMemcpy(m, m_cuda, Size * Size * sizeof(float),cudaMemcpyDeviceToHost );
	cudaMemcpy(a, a_cuda, Size * Size * sizeof(float),cudaMemcpyDeviceToHost );
	cudaMemcpy(b, b_cuda, Size * sizeof(float),cudaMemcpyDeviceToHost );
	cudaFree(m_cuda);
	cudaFree(a_cuda);
	cudaFree(b_cuda);
}
void BackSub()
{
	finalVec = (float *) malloc(Size * sizeof(float));
	int i,j;
	for(i=0;i<Size;i++){
		finalVec[Size-i-1]=b[Size-i-1];
		for(j=0;j<i;j++)
		{
			finalVec[Size-i-1]-=*(a+Size*(Size-i-1)+(Size-j-1)) * finalVec[Size-j-1];
		}
		finalVec[Size-i-1]=finalVec[Size-i-1]/ *(a+Size*(Size-i-1)+(Size-i-1));
	}
}