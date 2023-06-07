defmodule MyKernel do
  import GPotion

gpotion add_vectors(result, a, b, n, [:matrex,:matrex,:matrex,:int]) do
  var index int = threadIdx.x + blockIdx.x * blockDim.x;
  var stride int = blockDim.x * gridDim.x;
  for i in range(index,n,stride) do
         result[i] = a[i] + b[i]
  end
end
end

n = 10000000



list = [Enum.to_list(1..n)]

vet1 = Matrex.new(list)
vet2 = Matrex.new(list)


kernel=GPotion.load(&MyKernel.add_vectors/5)

threadsPerBlock = 128;
numberOfBlocks = div(n + threadsPerBlock - 1, threadsPerBlock)


prev = System.monotonic_time()

ref1=GPotion.new_gmatrex(vet1)
ref2=GPotion.new_gmatrex(vet2)
ref3=GPotion.new_gmatrex(1,n)

GPotion.spawn(kernel,{numberOfBlocks,1,1},{threadsPerBlock,1,1},[ref3,ref1,ref2,n])
GPotion.synchronize()

next = System.monotonic_time()
IO.puts "time gpu #{System.convert_time_unit(next-prev,:native,:millisecond)}"

result = GPotion.get_gmatrex(ref3)
#IO.inspect result


prev = System.monotonic_time()
eresult = Matrex.add(vet1,vet2)
next = System.monotonic_time()
IO.puts "time cpu #{System.convert_time_unit(next-prev,:native,:millisecond)}"

diff = Matrex.subtract(result,eresult)

IO.puts "this value must be zero: #{Matrex.sum(diff)}"
