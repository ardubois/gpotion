#include <stdio.h>
#include <sys/time.h>
#include <float.h>
#include <vector>
#include "cuda.h"
#include <time.h>
#define min( a, b )			a > b ? b : a
#define ceilDiv( a, b )		( a + b - 1 ) / b
#define print( x )			printf( #x ": %lu\n", (unsigned long) x )
#define DEFAULT_THREADS_PER_BLOCK 256
typedef struct latLong
{
  float lat;
  float lng;
} LatLong;
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
int main(int argc, char* argv[])
{
	std::vector<LatLong> locations;
  int numRecords = atoi(argv[1]);
   loadData(locations,numRecords);
	float *distances;
	LatLong *d_locations;
	float *d_distances;
 float time;
    cudaEvent_t start, stop;   
     cudaEventCreate(&start) ;
    cudaEventCreate(&stop) ;
    cudaEventRecord(start, 0) ;
	distances = (float *)malloc(sizeof(float) * numRecords);
	cudaMalloc((void **) &d_locations,sizeof(LatLong) * numRecords);
	cudaMalloc((void **) &d_distances,sizeof(float) * numRecords);
    cudaMemcpy( d_locations, &locations[0], sizeof(LatLong) * numRecords, cudaMemcpyHostToDevice);
    euclid<<< numRecords, 1 >>>(d_locations,d_distances,numRecords,lat,lng);
    cudaDeviceSynchronize();
    cudaMemcpy( distances, d_distances, sizeof(float) * numRecords, cudaMemcpyDeviceToHost );
    free(distances);
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
int loadDatafile(char *filename,std::vector<Record> &records,std::vector<LatLong> &locations){
    FILE   *flist,*fp;
	int    i=0;
	char dbname[64];
	int recNum=0;
    flist = fopen(filename, "r");
	while(!feof(flist)) {
		if(fscanf(flist, "%s\n", dbname) != 1) {
            fprintf(stderr, "error reading filelist\n");
            exit(0);
        }
        fp = fopen(dbname, "r");
        if(!fp) {
            printf("error opening a db\n");
            exit(1);
        }
        while(!feof(fp)){
            Record record;
            LatLong latLong;
            fgets(record.recString,49,fp);
            fgetc(fp); // newline
            if (feof(fp)) break;
            char substr[6];
            for(i=0;i<5;i++) substr[i] = *(record.recString+i+28);
            substr[5] = '\0';
            latLong.lat = atof(substr);
            for(i=0;i<5;i++) substr[i] = *(record.recString+i+33);
            substr[5] = '\0';
            latLong.lng = atof(substr);
            locations.push_back(latLong);
            records.push_back(record);
            recNum++;
        }
        fclose(fp);
    }
    fclose(flist);
    return recNum;
}