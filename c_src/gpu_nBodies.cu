#include "erl_nif.h"

__global__
void gpu_nBodies(float *p, float dt, int n, float softening)
{
	int i = ((blockDim.x * blockIdx.x) + threadIdx.x);
if((i < n))
{
	float fx = 0.0;
	float fy = 0.0;
	float fz = 0.0;
for( int j = 0; j<n; j++){
	float dx = (p[(6 * j)] - p[(6 * i)]);
	float dy = (p[((6 * j) + 1)] - p[((6 * i) + 1)]);
	float dz = (p[((6 * j) + 2)] - p[((6 * i) + 2)]);
	float distSqr = ((((dx * dx) + (dy * dy)) + (dz * dz)) + softening);
	float invDist = (1.0 / sqrt(distSqr));
	float invDist3 = ((invDist * invDist) * invDist);
	fx = (fx + (dx * invDist3));
	fy = (fy + (dy * invDist3));
	fz = (fz + (dz * invDist3));
}

	p[((6 * i) + 3)] = (p[((6 * i) + 3)] + (dt * fx));
	p[((6 * i) + 4)] = (p[((6 * i) + 4)] + (dt * fy));
	p[((6 * i) + 5)] = (p[((6 * i) + 5)] + (dt * fz));
}

}

extern "C" void gpu_nBodies_call(ErlNifEnv *env, const ERL_NIF_TERM argv[], ErlNifResourceType* type)
  {

    ERL_NIF_TERM list;
    ERL_NIF_TERM head;
    ERL_NIF_TERM tail;
    float **array_res;

    const ERL_NIF_TERM *tuple_blocks;
    const ERL_NIF_TERM *tuple_threads;
    int arity;

    if (!enif_get_tuple(env, argv[1], &arity, &tuple_blocks)) {
      printf ("spawn: blocks argument is not a tuple");
    }

    if (!enif_get_tuple(env, argv[2], &arity, &tuple_threads)) {
      printf ("spawn:threads argument is not a tuple");
    }
    int b1,b2,b3,t1,t2,t3;

    enif_get_int(env,tuple_blocks[0],&b1);
    enif_get_int(env,tuple_blocks[1],&b2);
    enif_get_int(env,tuple_blocks[2],&b3);
    enif_get_int(env,tuple_threads[0],&t1);
    enif_get_int(env,tuple_threads[1],&t2);
    enif_get_int(env,tuple_threads[2],&t3);

    dim3 blocks(b1,b2,b3);
    dim3 threads(t1,t2,t3);

    list= argv[3];

  enif_get_list_cell(env,list,&head,&tail);
  enif_get_resource(env, head, type, (void **) &array_res);
  float *arg1 = *array_res;
  list = tail;

  enif_get_list_cell(env,list,&head,&tail);
  double darg2;
  float arg2;
  enif_get_double(env, head, &darg2);
  arg2 = (float) darg2;
  list = tail;

  enif_get_list_cell(env,list,&head,&tail);
  int arg3;
  enif_get_int(env, head, &arg3);
  list = tail;

  enif_get_list_cell(env,list,&head,&tail);
  double darg4;
  float arg4;
  enif_get_double(env, head, &darg4);
  arg4 = (float) darg4;
  list = tail;

   gpu_nBodies<<<blocks, threads>>>(arg1,arg2,arg3,arg4);
    cudaError_t error_gpu = cudaGetLastError();
    if(error_gpu != cudaSuccess)
     { char message[200];
       strcpy(message,"Error kernel call: ");
       strcat(message, cudaGetErrorString(error_gpu));
       enif_raise_exception(env,enif_make_string(env, message, ERL_NIF_LATIN1));
     }
}
