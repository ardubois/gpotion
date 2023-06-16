

defmodule MM do
  import GPotion
gpotion mm(a,b,c,m,n,k) do
  row  = blockIdx.y * blockDim.y + threadIdx.y
  col = blockIdx.x * blockDim.x + threadIdx.x
  sum  = 0.0
  if(col < k && row < m) do
    for i in range(0,n,1) do
      sum = sum + a[row * n + i] * b[i * k + col]
    end
    c[row * k + col] = sum
  end

end
end

[arg] = System.argv()

m = String.to_integer(arg)
n = m
k=m




mat = Matrex.fill(1,m*k,1)

f = fn _ -> Enum.random(1..100) end

mat1 = Matrex.apply(mat,f)
mat2 = Matrex.apply(mat,f)


block_size = 16
grid_rows = trunc ((m + block_size - 1) / block_size)
grid_cols = trunc ((k + block_size - 1) / block_size)


prev = System.monotonic_time()
ker=GPotion.load(&MM.mm/6)
a=GPotion.new_gmatrex(mat1)
b=GPotion.new_gmatrex(mat2)
c=GPotion.new_gmatrex(1,m*k)

GPotion.spawn(ker,{grid_rows,grid_cols,1},{block_size,block_size,1},[a,b,c,m,n,k])
GPotion.synchronize()

_result = GPotion.get_gmatrex(c)

next = System.monotonic_time()
#IO.puts "time gpu #{System.convert_time_unit(next-prev,:native,:millisecond)}"
IO.puts "GPotion\t#{m}\t#{System.convert_time_unit(next-prev,:native,:millisecond)} "

#IO.inspect result
#IO.puts GPU.Backend.gen_c_kernel('addVectors',4,[])
