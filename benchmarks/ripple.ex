defmodule BMP do
  @on_load :load_nifs
  def load_nifs do
      :erlang.load_nif('./priv/bmp_nifs', 0)
  end
  def gen_bmp_nif(_string,_dim,_mat) do
      raise "gen_bmp_nif not implemented"
  end
  def gen_bmp(string,dim,%Matrex{data: matrix} = _a) do
    gen_bmp_nif(string,dim,matrix)
  end
end

defmodule Ripple do
  import GPotion
  gpotion ripple_kernel(ptr,dim,ticks) do
    x = threadIdx.x + blockIdx.x * blockDim.x;
    y = threadIdx.y + blockIdx.y * blockDim.y;
    offset = x + y * blockDim.x * gridDim.x;

    fx = 0.5 *  x - dim/15;
    fy = 0.5 *  y - dim/15;
    d = sqrtf( fx * fx + fy * fy );
    grey = floor(128.0 + 127.0 *cos(d/10.0 - ticks/7.0) /(d/10.0 + 1.0));
    ptr[offset*4 + 0] = grey;
    ptr[offset*4 + 1] = grey;
    ptr[offset*4 + 2] = grey;
    ptr[offset*4 + 3] = 255;
  end
end

[arg] = System.argv()

user_value = String.to_integer(arg)
dim =user_value #300

#mat = Matrex.fill(1,dim*dim*4,0)



ker=GPotion.load(&Ripple.ripple_kernel/4)
n_threads = 128
_n_blocks = floor ((dim+(n_threads-1))/n_threads)


prev = System.monotonic_time()

ref=GPotion.new_gmatrex(1,dim*dim*4)
GPotion.spawn(ker,{dim,dim,1},{1,1,1},[ref,dim,10])
GPotion.synchronize()
image = GPotion.get_gmatrex(ref)

next = System.monotonic_time()
IO.puts "GPotion\t#{dim}\t#{System.convert_time_unit(next-prev,:native,:millisecond)}"



BMP.gen_bmp('ripplegpu.bmp',dim,image)
