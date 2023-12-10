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
gptype julia integer ~> integer ~> integer ~> integer
gpdef  julia(x,y,dim) do
  var scale float = 0.1
  var jx float = scale * (dim - x)/dim
  var jy float = scale * (dim - y)/dim

  var cr float = -0.8
  var ci float = 0.156
  var ar float = jx
  var ai float = jy
  for i in range(0,200) do
      var nar float  = (ar*ar - ai*ai) + cr
      var nai float = (ai*ar + ar*ai) + ci
      if ((nar * nar)+(nai * nai ) > 1000) do
        return 0
      end
      ar = nar
      ai = nai
  end
  return 1
end
gptype  julia_kernel gmatrex ~> integer ~> unit
gpotion julia_kernel(ptr,dim) do
  var x int = blockIdx.x
  var y int = blockIdx.y
  var offset int = x + y * dim # gridDim.x
#####
  var juliaValue int = julia(x,y,dim)

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
