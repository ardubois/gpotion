#include "erl_nif.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <dlfcn.h>
#include <string.h>
#include <malloc.h>

#define _bitsperpixel 32
#define _planes 1
#define _compression 0
#define _xpixelpermeter 0x13B //0x130B //2835 , 72 DPI
#define _ypixelpermeter 0x13B//0x130B //2835 , 72 DPI

#pragma pack(push,1)
typedef struct{
    uint8_t signature[2];
    uint32_t filesize;
    uint32_t reserved;
    uint32_t fileoffset_to_pixelarray;
} fileheader;
typedef struct{
    uint32_t dibheadersize;
    uint32_t width;
    uint32_t height;
    uint16_t planes;
    uint16_t bitsperpixel;
    uint32_t compression;
    uint32_t imagesize;
    uint32_t ypixelpermeter;
    uint32_t xpixelpermeter;
    uint32_t numcolorspallette;
    uint32_t mostimpcolor;
} bitmapinfoheader;
typedef struct {
    fileheader fileheader;
    bitmapinfoheader bitmapinfoheader;
} bitmap;
#pragma pack(pop)

#define MX_ROWS(matrix) (((uint32_t*)matrix)[0])
#define MX_COLS(matrix) (((uint32_t*)matrix)[1])
#define MX_SET_ROWS(matrix, rows) ((uint32_t*)matrix)[0] = rows
#define MX_SET_COLS(matrix, cols) ((uint32_t*)matrix)[1] = cols
#define MX_LENGTH(matrix) ((((uint32_t*)matrix)[0])*(((uint32_t*)matrix)[1]) + 2)


void genBpm (uint32_t height, uint32_t width, uint8_t *pixelbuffer, char *file_name) {
    uint32_t pixelbytesize = height*width*_bitsperpixel/8;
    uint32_t  _filesize =pixelbytesize+sizeof(bitmap);
    FILE *fp = fopen(file_name,"wb");
    bitmap *pbitmap  = (bitmap*)calloc(1,sizeof(bitmap));


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
    fwrite(pixelbuffer,1,pixelbytesize,fp);
    fclose(fp);
    free(pbitmap);
    
}


static ERL_NIF_TERM gen_bmp_nif(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
  ErlNifBinary  matrix_el;
  float         *matrix;
  int dim;
  
  ///GET FILE NAME
  ERL_NIF_TERM list = argv[0];
 
  unsigned int size;
  enif_get_list_length(env,list,&size);
  char file_name[1024];
  
  enif_get_string(env,list,file_name,size+1,ERL_NIF_LATIN1);
 ///// END GET FILE NAME
  /// GET DIM
 
  if (!enif_get_int(env, argv[1], &dim)) {
      return enif_make_badarg(env);
  }
  //// END GET DIM
  /// BEGIN GET MATREX
  if (!enif_inspect_binary(env, argv[2], &matrix_el)) 
    {
       return enif_make_badarg(env);
    }
  matrix = (float *) matrix_el.data;
  
  matrix +=2; 
  //// END GET MATREX
  
  int matrex_size=dim*dim*4;

  uint8_t *pixelbuffer = (uint8_t*)malloc(matrex_size);

  for(int i = 0; i<matrex_size;i++)
  {
     pixelbuffer[i]= (uint8_t) matrix[i];
  }

  

 // printf("matrex size %d\n",matrex_size);

   // for(int i=0;i<matrex_size; i++)
     //  {   printf("matrex %d = %f\n",i,matrix[i]);
       //    printf("pixel %d = %d\n",i,pixelbuffer[i]);
       //}
           
  genBpm(dim,dim,pixelbuffer,file_name);
  //printf("size matrex %d, size image %d\n", data_size, dim*dim*4);
  
  free(pixelbuffer);
  return enif_make_int(env, 0);

}


static ErlNifFunc nif_funcs[] = {
    {"gen_bmp_nif", 3, gen_bmp_nif}
};

ERL_NIF_INIT(Elixir.BMP, nif_funcs, NULL, NULL, NULL, NULL)