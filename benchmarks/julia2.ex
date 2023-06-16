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

defmodule Julia do
  import GPotion
gpdef julia(x,y,dim) do
  scale = 0.1
  jx = scale * (dim - x)/dim
  jy = scale * (dim - y)/dim

  cr = -0.8
  ci = 0.156
  ar = jx
  ai = jy
  for i in range(0,200) do
      nar = (ar*ar - ai*ai) + cr
      nai = (ai*ar + ar*ai) + ci
      if ((nar * nar)+(nai * nai ) > 1000) do
        return 0
      end
      ar = nar
      ai = nai
  end
  return 1
end
gpotion julia_kernel(ptr,dim) do
  x = blockIdx.x
  y = blockIdx.y
  offset = x + y * dim # gridDim.x
#####
  juliaValue = julia(x,y,dim)

  #if (juliaValue != 0) do
  #  juliaValue = 1
  #end
#####
  ptr[offset*4 + 0] = 255 * juliaValue;
  ptr[offset*4 + 1] = 0;
  ptr[offset*4 + 2] = 0;
  ptr[offset*4 + 3] = 255;

end
end

[arg] = System.argv()
m = String.to_integer(arg)

dim = m


ker=GPotion.load(&Julia.julia_kernel/3)

prev = System.monotonic_time()
ref=GPotion.new_gmatrex(1,dim*dim*4)
GPotion.spawn(ker,{dim,dim,1},{1,1,1},[ref,dim])
GPotion.synchronize()
image = GPotion.get_gmatrex(ref)
next = System.monotonic_time()

IO.puts "GPotion\t#{dim}\t#{System.convert_time_unit(next-prev,:native,:millisecond)}"



BMP.gen_bmp('julia2gpotion.bmp',dim,image)
