#include "erl_nif.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <dlfcn.h>


#define MX_ROWS(matrix) (((uint32_t*)matrix)[0])
#define MX_COLS(matrix) (((uint32_t*)matrix)[1])
#define MX_SET_ROWS(matrix, rows) ((uint32_t*)matrix)[0] = rows
#define MX_SET_COLS(matrix, cols) ((uint32_t*)matrix)[1] = cols
#define MX_LENGTH(matrix) ((((uint32_t*)matrix)[0])*(((uint32_t*)matrix)[1]) + 2)


ErlNifResourceType *KERNEL_TYPE;
ErlNifResourceType *ARRAY_TYPE;
ErlNifResourceType *PINNED_ARRAY;

void
dev_array_destructor(ErlNifEnv *env, void *res) {
  float **dev_array = (float**) res;
  cudaFree(*dev_array);
}

void
dev_pinned_array_destructor(ErlNifEnv *env, void *res) {
  float **dev_array = (float**) res;
  cudaFreeHost(*dev_array);
}

static int
load(ErlNifEnv *env, void **priv_data, ERL_NIF_TERM load_info) {
  KERNEL_TYPE =
  enif_open_resource_type(env, NULL, "kernel", NULL, ERL_NIF_RT_CREATE  , NULL);
  ARRAY_TYPE =
  enif_open_resource_type(env, NULL, "gpu_ref", dev_array_destructor, ERL_NIF_RT_CREATE  , NULL);
  PINNED_ARRAY =
  enif_open_resource_type(env, NULL, "pinned_array", dev_pinned_array_destructor, ERL_NIF_RT_CREATE  , NULL);
  return 0;
}

static ERL_NIF_TERM new_pinned_nif(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {

  float *host_matrix;
  cudaError_t error_gpu;
  int length;
  ERL_NIF_TERM list;
  ERL_NIF_TERM head;
  ERL_NIF_TERM tail;
  if (!enif_get_list_cell(env,argv[0],&head,&tail)) return enif_make_badarg(env);
  if (!enif_get_int(env, argv[1], &length))  return enif_make_badarg(env);




  int data_size = sizeof(float)*(length+2);
  

  ///// MAKE CUDA CALL
  cudaMallocHost( (void**)&host_matrix, data_size);
  error_gpu = cudaGetLastError();
  if(error_gpu != cudaSuccess)  
      { char message[200];
        strcpy(message,"Error new_pinned_nif: ");
        strcat(message, cudaGetErrorString(error_gpu));
        enif_raise_exception(env,enif_make_string(env, message, ERL_NIF_LATIN1));
      }

  MX_SET_ROWS(host_matrix, 1);
  MX_SET_COLS(host_matrix, length);

  list = argv[0];

  for(int i=2;i<(length+2);i++)
  {
    enif_get_list_cell(env,list,&head,&tail);
    double dvalue;
    enif_get_double(env, head, &dvalue);
    host_matrix[i] = (float) dvalue;
    list = tail;

  }


  float **pinned_res = (float**)enif_alloc_resource(PINNED_ARRAY, sizeof(float *));
  *pinned_res = host_matrix;
  ERL_NIF_TERM term = enif_make_resource(env, pinned_res);
  // ...and release the resource so that it will be freed when Erlang garbage collects
  enif_release_resource(pinned_res);

  return term;

}
static ERL_NIF_TERM create_ref_nif(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
  ErlNifBinary  matrix_el;
  float         *matrix;
  float         *dev_matrix;
  cudaError_t error_gpu;
  
  if (!enif_inspect_binary(env, argv[0], &matrix_el)) return enif_make_badarg(env);

  matrix = (float *) matrix_el.data;
  uint64_t data_size = sizeof(float)*(MX_LENGTH(matrix)-2);
  
  matrix +=2; 

  ///// MAKE CUDA CALL
  cudaMalloc( (void**)&dev_matrix, data_size);
  error_gpu = cudaGetLastError();
  if(error_gpu != cudaSuccess)  
      { char message[200];
        strcpy(message,"Error create_ref_nif: ");
        strcat(message, cudaGetErrorString(error_gpu));
        enif_raise_exception(env,enif_make_string(env, message, ERL_NIF_LATIN1));
      }
  ///// MAKE CUDA CALL
  cudaMemcpy( dev_matrix, matrix, data_size, cudaMemcpyHostToDevice );
  error_gpu = cudaGetLastError();
  if(error_gpu != cudaSuccess)  
      { char message[200];
        strcpy(message,"Error create_ref_nif: ");
        strcat(message, cudaGetErrorString(error_gpu));
        enif_raise_exception(env,enif_make_string(env, message, ERL_NIF_LATIN1));
      }
  
  /////////// END CUDA CALL

  float **gpu_res = (float**)enif_alloc_resource(ARRAY_TYPE, sizeof(float *));
  *gpu_res = dev_matrix;
  ERL_NIF_TERM term = enif_make_resource(env, gpu_res);
  // ...and release the resource so that it will be freed when Erlang garbage collects
  enif_release_resource(gpu_res);

  return term;
}

static ERL_NIF_TERM new_ref_nif(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
  float         *dev_matrix;
  int data_size;
  cudaError_t error_gpu;
  
  if (!enif_get_int(env, argv[0], &data_size)) {
      return enif_make_badarg(env);
  }
 
  data_size = data_size * sizeof(float);

  //// MAKE CUDA CALL
  cudaMalloc( (void**)&dev_matrix, data_size);
  error_gpu = cudaGetLastError();
  if(error_gpu != cudaSuccess)  
      { char message[200];
        strcpy(message,"Error new_ref_nif: ");
        strcat(message, cudaGetErrorString(error_gpu));
        enif_raise_exception(env,enif_make_string(env, message, ERL_NIF_LATIN1));
      }

  //END CUDA CALL


  float **gpu_res = (float**)enif_alloc_resource(ARRAY_TYPE, sizeof(float *));
  *gpu_res = dev_matrix;
  ERL_NIF_TERM term = enif_make_resource(env, gpu_res);
  // ...and release the resource so that it will be freed when Erlang garbage collects
  enif_release_resource(gpu_res);

  return term;
}


static ERL_NIF_TERM get_matrex_nif(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
  int nrow;
  int ncol;
  ERL_NIF_TERM  result;
  float **array_res;
  cudaError_t error_gpu;
  
  if (!enif_get_resource(env, argv[0], ARRAY_TYPE, (void **) &array_res)) {
    return enif_make_badarg(env);
  }
  
  if (!enif_get_int(env, argv[1], &nrow)) {
      return enif_make_badarg(env);
  }
  
  if (!enif_get_int(env, argv[2], &ncol)) {
      return enif_make_badarg(env);
  }
  
  float *dev_array = *array_res;

  int result_size = sizeof(float) * (nrow*ncol+2);
  int data_size = sizeof(float) * (nrow*ncol);
  float *result_data = (float *) enif_make_new_binary(env, result_size, &result);

  float *ptr_matrix ;
  ptr_matrix = result_data;
  ptr_matrix +=2;


  //// MAKE CUDA CALL
  cudaMemcpy(ptr_matrix, dev_array, data_size, cudaMemcpyDeviceToHost );
  error_gpu = cudaGetLastError();
  if(error_gpu != cudaSuccess)  
      { char message[200];
        strcpy(message,"Error get_matrex_nif: ");
        strcat(message, cudaGetErrorString(error_gpu));
        enif_raise_exception(env,enif_make_string(env, message, ERL_NIF_LATIN1));
      }
  //////// END CUDA CALL

  MX_SET_ROWS(result_data, nrow);
  MX_SET_COLS(result_data, ncol);
  
  return result;
}

static ERL_NIF_TERM synchronize_nif(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
  cudaError_t error_gpu;

  ////// MAKE CUDA CALL
  cudaDeviceSynchronize();
  error_gpu = cudaGetLastError();
  if(error_gpu != cudaSuccess)  
      { char message[200];
        strcpy(message,"Error synchronize_nif: ");
        strcat(message, cudaGetErrorString(error_gpu));
        enif_raise_exception(env,enif_make_string(env, message, ERL_NIF_LATIN1));
      }
  //// END CUDA CALL
  return enif_make_int(env, 0);
}

static ERL_NIF_TERM load_kernel_nif(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
   
  ERL_NIF_TERM e_name_module = argv[0];
  ERL_NIF_TERM e_name_fun = argv[1];
  
  unsigned int size_name_module;
  unsigned int size_name_fun;
  

  enif_get_list_length(env,e_name_fun,&size_name_fun);
  enif_get_list_length(env,e_name_module,&size_name_module);

  char kernel_name[1024];
  char func_name[1024];
  char lib_name[1024];
  char module_name[1024];

  enif_get_string(env,e_name_fun,kernel_name,size_name_fun+1,ERL_NIF_LATIN1);
  enif_get_string(env,e_name_module,module_name,size_name_module+1,ERL_NIF_LATIN1);

  strcpy(func_name,kernel_name);
  strcpy(lib_name,"priv/");
  strcat(lib_name,module_name);
  strcat(func_name,"_call");
  strcat(lib_name,".so");

  //strcpy(func_name, "print");

 // printf("libname %s\n",lib_name);
  
  void * m_handle = dlopen(lib_name, RTLD_NOW);
  if(m_handle== NULL)  
      { char message[200];
        strcpy(message,"Error opening dll!! ");
        enif_raise_exception(env,enif_make_string(env, message, ERL_NIF_LATIN1));
      }



  void (*fn)();
  fn= (void (*)())dlsym( m_handle, func_name);
  
  void (**kernel_res)() = (void (**)()) enif_alloc_resource(KERNEL_TYPE, sizeof(void *));

  // Let's create conn and let the resource point to it
  
  *kernel_res = fn;
  
  // We can now make the Erlang term that holds the resource...
  ERL_NIF_TERM term = enif_make_resource(env, kernel_res);
  // ...and release the resource so that it will be freed when Erlang garbage collects
  enif_release_resource(kernel_res);
 

  return term;
}

static ERL_NIF_TERM spawn_nif(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
  void (**kernel_res)(ErlNifEnv *env, const ERL_NIF_TERM argv[], ErlNifResourceType* type);
  //void (**kernel_res)();
  //float **array_res;
  //printf("spawn begin\n");
  //fflush(stdout);
  if (!enif_get_resource(env, argv[0], KERNEL_TYPE, (void **) &kernel_res)) {
    return enif_make_badarg(env);
  }
  
  void (*fn)(ErlNifEnv *env, const ERL_NIF_TERM argv[], ErlNifResourceType* type) = *kernel_res;
  //void (*fn)() = *kernel_res;
  //float *array = *array_res;
  //printf("ok nif");
  (*fn)(env,argv,ARRAY_TYPE);
  //(*fn)();



  return enif_make_int(env, 0);
}

static ErlNifFunc nif_funcs[] = {
    {"load_kernel_nif", 2, load_kernel_nif},
    {"new_pinned_niv",2,new_pinned_nif},
    {"spawn_nif", 4,spawn_nif},
    {"create_ref_nif", 1, create_ref_nif},
    {"new_ref_nif", 1, new_ref_nif},
    {"get_matrex_nif", 3, get_matrex_nif},
    {"synchronize_nif", 0, synchronize_nif}
};

ERL_NIF_INIT(Elixir.GPotion, nif_funcs, &load, NULL, NULL, NULL)
