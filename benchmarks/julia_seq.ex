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

def julia_seq(dim, [{y,x}]) do
  gen_pixel(dim,{y,x})
end
def julia_seq(dim,[{y,x}|tail]) do
  pixel = gen_pixel(dim,{y,x})
  pixels=julia_seq(dim,tail)
  pixel ++ pixels
end
def gen_pixel(dim,{y,x}) do
  juliaValue = julia( x, y, dim )
  [255 * juliaValue,0,0,255]
end
def julia(x,y,dim) do
  scale = 0.1
  jx = scale * (dim - x)/dim
  jy = scale * (dim - y)/dim
  cr  = -0.8
  ci  = 0.156
  ar  = jx
  ai  = jy
  test_julia(200,cr,ci,ar,ai,1000)
end
def test_julia(0,_cr,_ci,_ar,_ai,_number) do
  1
end
def test_julia(n,cr,ci,ar,ai,number) do
  nar = ((ar * ar) - (ai*ai)) + cr
  nai = ((ai * ar) + (ar*ai)) + ci
  if (nar*nar + nai*nai > number) do
    0
  else
    test_julia(n-1,cr,ci,nar,nai,number)
  end
end
end


[arg] = System.argv()

user_value = String.to_integer(arg)

dim = user_value


prev = System.monotonic_time()
indices = for i <- Enum.to_list(0..(dim-1)), j<-Enum.to_list(0..(dim-1)), do: {i,j}
imageseq = Julia.julia_seq(dim,indices)
next = System.monotonic_time()
IO.puts "time cpu #{System.convert_time_unit(next-prev,:native,:millisecond)}"
imageseq=Matrex.new([imageseq])
BMP.gen_bmp('julia_seq.bmp',dim,imageseq)
