#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <malloc.h>

__global__
void julia_kernel(float *ptr, int dim)
{
int x = blockIdx.x;
int y = blockIdx.y;
int offset = (x + (y * dim));
int juliaValue = 1;
float scale = 0.1;
float jx = ((scale * (dim - x)) / dim);
float jy = ((scale * (dim - y)) / dim);
float cr = (- 0.8);
float ci = 0.156;
float ar = jx;
float ai = jy;
for( int i = 0; i<200; i++){
float nar = (((ar * ar) - (ai * ai)) + cr);
float nai = (((ai * ar) + (ar * ai)) + ci);
if((((nar * nar) + (nai * nai)) > 1000))
{
	juliaValue = 0;
break;
}

	ar = nar;
	ai = nai;
}

	ptr[((offset * 4) + 0)] = (255 * juliaValue);
	ptr[((offset * 4) + 1)] = 0;
	ptr[((offset * 4) + 2)] = 0;
	ptr[((offset * 4) + 3)] = 255;
}



void genBpm (int height, int width, float *pixelbuffer_f) {
    uint32_t pixelbytesize = height*width*_bitsperpixel/8;
    uint32_t  _filesize =pixelbytesize+sizeof(bitmap);
    FILE *fp = fopen("test.bmp","wb");
    bitmap *pbitmap  = (bitmap*)calloc(1,sizeof(bitmap));

    int buffer_size = height*width*4;
    uint8_t *pixelbuffer = (uint8_t*)malloc(buffer_size);

    for(int i = 0; i<buffer_size;i++)
    {
     pixelbuffer[i]= (uint8_t) pixelbuffer_f[i];
    }


    //strcpy(pbitmap->fileheader.signature,"BM");
    pbitmap->fileheader.signature[0] = 'B';
    pbitmap->fileheader.signature[1] = 'M';
    pbitmap->fileheader.filesize = _filesize;
    pbitmap->fileheader.fileoffset_to_pixelarray = sizeof(bitmap);
    pbitmap->bitmapinfoheader.dibheadersize =sizeof(bitmapinfoheader);
    pbitmap->bitmapinfoheader.width = width;
    pbitmap->bitmapinfoheader.height = height;
    pbitmap->bitmapinfoheader.planes = _planes;
    pbitmap->bitmapinfoheader.bitsperpixel = _bitsperpixel;
    pbitmap->bitmapinfoheader.compression = _compression;
    pbitmap->bitmapinfoheader.imagesize = pixelbytesize;
    pbitmap->bitmapinfoheader.ypixelpermeter = _ypixelpermeter ;
    pbitmap->bitmapinfoheader.xpixelpermeter = _xpixelpermeter ;
    pbitmap->bitmapinfoheader.numcolorspallette = 0;
    fwrite (pbitmap, 1, sizeof(bitmap),fp);
    //memset(pixelbuffer,pixel,pixelbytesize);
    fwrite(pixelbuffer,1,pixelbytesize,fp);
    fclose(fp);
    free(pbitmap);
    free(pixelbuffer);
}


int main( int argc, char const *argv[] ) {

    int usr_value = atoi(argv[1]);
   
    int height = usr_value;
    int width  = usr_value;
    int DIM = usr_value;
    int size_array = height*width*4*sizeof(float);
    cudaError_t j_error;
    
    //int pixelbytesize=  height*width*_bitsperpixel/8;
    //printf(" pixel byte size %lu\n",pixelbytesize);
   
    float time;
    cudaEvent_t start, stop;   
     cudaEventCreate(&start) ;
    cudaEventCreate(&stop) ;
    cudaEventRecord(start, 0) ;


     float *h_pixelbuffer = (float*)malloc(size_array);
     float *d_pixelbuffer;

     ////////
    cudaMalloc( (void**)&d_pixelbuffer, size_array);
    j_error = cudaGetLastError();
    if(j_error != cudaSuccess) printf("Error 1: %s\n", cudaGetErrorString(j_error));
    ////////

    
    ////////////////////
    dim3 grid(DIM,DIM);

    julia_kernel<<<grid, 1>>>(d_pixelbuffer,height); // compute interbody forces
    j_error = cudaGetLastError();
    if(j_error != cudaSuccess) printf("Error 3: %s\n", cudaGetErrorString(j_error));
  ////////



    cudaMemcpy(h_pixelbuffer, d_pixelbuffer, size_array, cudaMemcpyDeviceToHost); // return results 
    j_error = cudaGetLastError();
    if(j_error != cudaSuccess) printf("Error 7: %s\n", cudaGetErrorString(j_error));



    cudaEventRecord(stop, 0) ;
    cudaEventSynchronize(stop) ;
    cudaEventElapsedTime(&time, start, stop) ;

    printf("CUDA\t%d\t%3.1f\n", usr_value,time);
    
    genBpm(height,width,h_pixelbuffer);
   
    free(h_pixelbuffer);
    cudaFree(d_pixelbuffer);
}



