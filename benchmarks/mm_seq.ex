

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
_n = m
k=m




mat = Matrex.fill(1,m*k,1)

f = fn _ -> Enum.random(1..100) end

mat1 = Matrex.apply(mat,f)
mat2 = Matrex.apply(mat,f)


amat = Matrex.reshape(mat1,m,k)
bmat = Matrex.reshape(mat2,m,k)

prev = System.monotonic_time()
_cmat = Matrex.dot(amat,bmat)
next = System.monotonic_time()
#IO.puts "time cpu #{System.convert_time_unit(next-prev,:native,:millisecond)}"
IO.puts "Elixir\t#{m}\t#{System.convert_time_unit(next-prev,:native,:millisecond)} "

#rmat = Matrex.reshape(result,m,k)

#fmat = Matrex.subtract(cmat,rmat)


#IO.puts "this value must be zero: #{Matrex.sum(fmat)}"
