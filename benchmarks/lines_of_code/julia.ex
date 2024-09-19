defmodule Julia do
  import GPotion
gpotion julia_kernel(ptr,dim) do
  x = blockIdx.x
  y = blockIdx.y
  offset = x + y * dim # gridDim.x
  juliaValue = 1
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
        juliaValue = 0
        break
      end
      ar = nar
      ai = nai
  end
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
