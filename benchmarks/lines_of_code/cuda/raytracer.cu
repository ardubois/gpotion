#include <stdio.h>
#include <time.h>
#include <stdint.h>
#define rnd( x ) (x * rand() / RAND_MAX)
#define INF     2e10f
struct Sphere {
    float   r,b,g;
    float   radius;
    float   x,y,z;
};
void loadSpheres(Sphere *vet, int size, float dim, float radius, float sum){
	for (int i=0;i<size;i++){
			Sphere sphere;
            sphere.r = rnd(1.0);
            sphere.b = rnd(1.0);
            sphere.g = rnd(1.0);
            sphere.radius = rnd(radius) + sum;
            sphere.x = rnd(dim) - trunc(dim / 2);
            sphere.y = rnd(dim) - trunc(dim / 2);
            sphere.z = rnd(256) - 128;
            vet[i] = sphere;
           }
}
#define SPHERES 20
__global__ void kernel(int dim, Sphere * s,  float *ptr ) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;
    float   ox = (x - dim/2);
    float   oy = (y - dim/2);
    float   maxz = -99999;
    for(int i=0; i<SPHERES; i++) {
        float   n;
        float   t = -99999;
        float dx = ox - s[i].x;
        float dy = oy - s[i].y;
        float dz;
        if (dx*dx + dy*dy < s[i].radius * s[i].radius) {
            dz = sqrtf( s[i].radius * s[i].radius - dx*dx - dy*dy );
            n = dz / sqrtf( s[i].radius * s[i].radius );
            t = dz + s[i].z;

        } else {
            t = -99999;
        }
        if (t > maxz) {
              float fscale = n;
              r = s[i].r * fscale;
              g = s[i].g * fscale;
              b = s[i].b * fscale;
              maxz = t;
        }
    }
    ptr[offset*4 + 0] = (r * 255);
    ptr[offset*4 + 1] = (g * 255);
    ptr[offset*4 + 2] = (b * 255);
    ptr[offset*4 + 3] = 255;
}
int main(int argc, char *argv[]){
    int dim = atoi(argv[1]);
    float   *final_image;
    float   *dev_image;
    Sphere * s;
    final_image = (float*) malloc(dim * dim * sizeof(float)*4);
    Sphere *temp_s = (Sphere*)malloc( sizeof(Sphere) * SPHERES );
   loadSpheres(temp_s, SPHERES, dim, 160, 20);
    float time;
    cudaEvent_t start, stop;   
    cudaEventCreate(&start) ;
    cudaEventCreate(&stop) ;
    cudaEventRecord(start, 0) ;
    cudaMalloc( (void**)&dev_image, dim * dim * sizeof(float)*4);
    cudaMalloc( (void**)&s, sizeof(Sphere) * SPHERES );
   cudaMemcpy( s, temp_s, sizeof(Sphere) * SPHERES, cudaMemcpyHostToDevice );
    dim3    grids(dim/16,dim/16);
    dim3    threads(16,16);
    kernel<<<grids,threads>>>(dim, s, dev_image);
    cudaMemcpy( final_image, dev_image, dim * dim * sizeof(float) * 4,cudaMemcpyDeviceToHost );
    cudaFree( dev_image);
    cudaFree( s );   
    cudaEventRecord(stop, 0) ;
    cudaEventSynchronize(stop) ;
    cudaEventElapsedTime(&time, start, stop) ;
     printf("CUDA\t%d\t%3.1f\n", dim,time);
    free(temp_s);
    free(final_image);
}