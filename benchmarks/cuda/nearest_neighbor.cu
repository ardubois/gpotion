/*
 * nn.cu
 * Nearest Neighbor
 * Modified by André Du Bois: changed depracated api, creating data set in memory. clean up code not used
 */

#include <stdio.h>
#include <sys/time.h>
#include <float.h>
#include <vector>
#include "cuda.h"

#include <time.h>

#define min( a, b )			a > b ? b : a
#define ceilDiv( a, b )		( a + b - 1 ) / b
#define print( x )			printf( #x ": %lu\n", (unsigned long) x )
#define DEBUG				false

#define DEFAULT_THREADS_PER_BLOCK 256

#define MAX_ARGS 10
#define REC_LENGTH 53 // size of a record in db
#define LATITUDE_POS 28	// character position of the latitude value in each record
#define OPEN 10000	// initial value of nearest neighbors


typedef struct latLong
{
  float lat;
  float lng;
} LatLong;

typedef struct record
{
  char recString[REC_LENGTH];
  float distance;
} Record;

void loadData(std::vector<LatLong> &locations, int size);
//void findLowest(std::vector<Record> &records,float *distances,int numRecords,int topN);
//void printUsage();
//int parseCommandline(int argc, char *argv[], char* filename,int *r,float *lat,float *lng,
//                     int *q, int *t, int *p, int *d);

/**
* Kernel
* Executed on GPU
* Calculates the Euclidean distance from each record in the database to the target position
*/
__global__ void euclid(LatLong *d_locations, float *d_distances, int numRecords,float lat, float lng)
{
	//int globalId = gridDim.x * blockDim.x * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;
	int globalId = blockDim.x * ( gridDim.x * blockIdx.y + blockIdx.x ) + threadIdx.x; // more efficient
    LatLong *latLong = d_locations+globalId;
    if (globalId < numRecords) {
        float *dist=d_distances+globalId;
        *dist = (float)sqrt((lat-latLong->lat)*(lat-latLong->lat)+(lng-latLong->lng)*(lng-latLong->lng));
	}
}

/**
* This program finds the k-nearest neighbors
**/

int main(int argc, char* argv[])
{
//	int    i=0;
	float lat=0, lng=0;
	
  //  std::vector<Record> records;
	std::vector<LatLong> locations;

  int numRecords = atoi(argv[1]);
    

   // int numRecords = loadData(filename,records,locations);
   loadData(locations,numRecords);

    
	float *distances;
	//Pointers to device memory
	LatLong *d_locations;
	float *d_distances;


	// Scaling calculations - added by Sam Kauffman
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties( &deviceProp, 0 );
	 cudaDeviceSynchronize();
	unsigned long maxGridX = deviceProp.maxGridSize[0];
	unsigned long threadsPerBlock = min( deviceProp.maxThreadsPerBlock, DEFAULT_THREADS_PER_BLOCK );
	size_t totalDeviceMemory;
	size_t freeDeviceMemory;
	cudaMemGetInfo(  &freeDeviceMemory, &totalDeviceMemory );
	 cudaDeviceSynchronize();
	unsigned long usableDeviceMemory = freeDeviceMemory * 85 / 100; // 85% arbitrary throttle to compensate for known CUDA bug
	unsigned long maxThreads = usableDeviceMemory / 12; // 4 bytes in 3 vectors per thread
	if ( numRecords > maxThreads )
	{
		fprintf( stderr, "Error: Input too large.\n" );
		exit( 1 );
	}
	unsigned long blocks = ceilDiv( numRecords, threadsPerBlock ); // extra threads will do nothing
	unsigned long gridY = ceilDiv( blocks, maxGridX );
	unsigned long gridX = ceilDiv( blocks, gridY );
	// There will be no more than (gridY - 1) extra blocks
	dim3 gridDim( gridX, gridY );

	/*if ( DEBUG )
	{
		print( totalDeviceMemory ); // 804454400
		print( freeDeviceMemory );
		print( usableDeviceMemory );
		print( maxGridX ); // 65535
		print( deviceProp.maxThreadsPerBlock ); // 1024
		print( threadsPerBlock );
		print( maxThreads );
		print( blocks ); // 130933
		print( gridY );
		print( gridX );
	}*/

	/**
	* Allocate memory on host and device

  */

  float time;
    cudaEvent_t start, stop;   
     cudaEventCreate(&start) ;
    cudaEventCreate(&stop) ;
    cudaEventRecord(start, 0) ;

	distances = (float *)malloc(sizeof(float) * numRecords);
	cudaMalloc((void **) &d_locations,sizeof(LatLong) * numRecords);
	cudaMalloc((void **) &d_distances,sizeof(float) * numRecords);

   /**
    * Transfer data from host to device
    */
    cudaMemcpy( d_locations, &locations[0], sizeof(LatLong) * numRecords, cudaMemcpyHostToDevice);

    /**
    * Execute kernel
    */

    euclid<<< gridDim, threadsPerBlock >>>(d_locations,d_distances,numRecords,lat,lng);
    cudaDeviceSynchronize();


    //Copy data from device memory to host memory
    cudaMemcpy( distances, d_distances, sizeof(float)*numRecords, cudaMemcpyDeviceToHost );


	// find the resultsCount least distances
    free(distances);
    //Free memory
	cudaFree(d_locations);
	cudaFree(d_distances);
   
     cudaEventRecord(stop, 0) ;
    cudaEventSynchronize(stop) ;
    cudaEventElapsedTime(&time, start, stop) ;

     printf("CUDA\t%d\t%3.1f\n", numRecords,time);

}

void loadData(std::vector<LatLong> &locations, int size){
   
	for (int i=0;i<size;i++){
			LatLong latLong;
            latLong.lat = ((float)(7 + rand() % 63)) + ((float) rand() / (float) 0x7fffffff);

            latLong.lng = ((float)(rand() % 358)) + ((float) rand() / (float) 0x7fffffff); 

            locations.push_back(latLong);
            
           
        }
     
}



