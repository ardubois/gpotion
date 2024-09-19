#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
typedef struct { float x, y, z, vx, vy, vz; } Body;
void randomizeBodies(float *data, int n) {
  for (int i = 0; i < n; i++) {
    data[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
  }
}
__global__ void bodyForce(Body *p, float dt, int n,float softening) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n) {
    float Fx = 0.0f; float Fy = 0.0f; float Fz = 0.0f;
    for (int j = 0; j < n; j++) {
      float dx = p[j].x - p[i].x;
      float dy = p[j].y - p[i].y;
      float dz = p[j].z - p[i].z;
      float distSqr = dx*dx + dy*dy + dz*dz + softening;
      float invDist = 1.0/sqrt(distSqr);
      float invDist3 = invDist * invDist * invDist;
      Fx += dx * invDist3; Fy += dy * invDist3; Fz += dz * invDist3;
    }
    p[i].vx += dt*Fx; p[i].vy += dt*Fy; p[i].vz += dt*Fz;
  }
}
__global__ void gpu_bodyForce(float *p, float dt, int n, float softening) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n) {
    float Fx = 0.0f; float Fy = 0.0f; float Fz = 0.0f;
    for (int j = 0; j < n; j++) {
      float dx = p[6*j] - p[6*i];
      float dy = p[6*j+1] - p[6*i+1];
      float dz = p[6*j+2] - p[6*i+2];
      float distSqr = dx*dx + dy*dy + dz*dz + softening;
      float invDist = 1.0/sqrt(distSqr);
      float invDist3 = invDist * invDist * invDist;
      Fx += dx * invDist3; 
      Fy += dy * invDist3; 
      Fz += dz * invDist3;
    }
    p[6*i+3]+= dt*Fx; 
    p[6*i+4] += dt*Fy; 
    p[6*i+5] += dt*Fz;
  }
}
__global__ void gpu_integrate(float *p, float dt, int n) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n) {
      p[6*i] += p[6*i+3]*dt;
      p[6*i+1] += p[6*i+4]*dt;
      p[6*i+2] += p[6*i+5]*dt;
  }
}
int main(const int argc, const char** argv) {
  int user_value = atoi(argv[1]);
  int nBodies = user_value;
  int block_size =  128;
  float softening = 0.000000001;
  cudaError_t nb_error;
  const float dt = 0.01; // time step
  int bytes = nBodies*sizeof(Body);
  float *h_buf = (float*)malloc(bytes);
  float *d_resp = (float*)malloc(bytes);
  randomizeBodies(h_buf, 6*nBodies); // Init pos / vel data
  float *d_buf;
  int nBlocks = (nBodies + block_size - 1) / block_size;
    float time;
    cudaEvent_t start, stop;   
    cudaEventCreate(&start) ;
    cudaEventCreate(&stop) ;
    cudaEventRecord(start, 0) ;
  cudaMalloc(&d_buf, bytes);
  nb_error = cudaGetLastError();
    if(nb_error != cudaSuccess) printf("Error 1: %s\n", cudaGetErrorString(nb_error));
  cudaMemcpy(d_buf, h_buf, bytes, cudaMemcpyHostToDevice);
   nb_error = cudaGetLastError();
    if(nb_error != cudaSuccess) printf("Error 2: %s\n", cudaGetErrorString(nb_error));
    gpu_bodyForce<<<nBlocks, block_size>>>(d_buf, dt, nBodies,softening); // compute interbody forces
    nb_error = cudaGetLastError();
    if(nb_error != cudaSuccess) printf("Error 3: %s\n", cudaGetErrorString(nb_error));
    cudaDeviceSynchronize();
    nb_error = cudaGetLastError();
    if(nb_error != cudaSuccess) printf("Error 3: %s\n", cudaGetErrorString(nb_error));
   gpu_integrate<<<nBlocks, block_size>>>(d_buf, dt, nBodies); // compute interbody forces
    nb_error = cudaGetLastError();
    if(nb_error != cudaSuccess) printf("Error 3: %s\n", cudaGetErrorString(nb_error));
   cudaMemcpy(d_resp, d_buf, bytes, cudaMemcpyDeviceToHost);
    nb_error = cudaGetLastError();
    if(nb_error != cudaSuccess) printf("Error 4: %s\n", cudaGetErrorString(nb_error));
    cudaEventRecord(stop, 0) ;
    cudaEventSynchronize(stop) ;
    cudaEventElapsedTime(&time, start, stop) ;
    printf("cuda\t%d\t%3.1f\n", nBodies,time);
    free(h_buf);
    free(d_resp);
    cudaFree(d_buf);
}
