defmodule GPUDP do
  import GPotion
  gpotion dot_product(ref4, a, n) do

  __shared__ cache[256]

  tid = threadIdx.x + blockIdx.x * blockDim.x;
  cacheIndex = threadIdx.x
  temp = 0.0

  while (tid < n) do
    temp = a[tid]  + temp
    tid = blockDim.x * gridDim.x + tid
  end

  cache[cacheIndex] = temp
  __syncthreads()

  i = blockDim.x/2
  while (i != 0) do
    if (cacheIndex < i) do
      cache[cacheIndex] = cache[cacheIndex + i] + cache[cacheIndex]
    end
    __syncthreads()
    i = i/2
  end

  if (cacheIndex == 0) do
    ref4[blockIdx.x] = cache[0]
  end

end
end

{n, _} = Integer.parse(Enum.at(System.argv, 0))

#list = [Enum.to_list(0..(n-1))]

vet1 = Matrex.new((1, n, fn -> :rand.uniform() end)
#vet2 = Matrex.new(list)


threadsPerBlock = 256
blocksPerGrid = div(n + threadsPerBlock - 1, threadsPerBlock)
numberOfBlocks = blocksPerGrid


prev = System.monotonic_time()

kernel=GPotion.load(&GPUDP.dot_product/4)

ref1=GPotion.new_gmatrex(vet1)
#ref2=GPotion.new_gmatrex(vet2)
ref3=GPotion.new_gmatrex(1,blocksPerGrid)

GPotion.spawn(kernel,{numberOfBlocks,1,1},{threadsPerBlock,1,1},[ref3, ref1,n])
GPotion.synchronize()

resultreal = GPotion.get_gmatrex(ref3)
_s = Matrex.sum(resultreal)
next = System.monotonic_time()

IO.puts "GPotion\t#{n}\t#{System.convert_time_unit(next-prev,:native,:millisecond)}"
