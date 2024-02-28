# fatoração LU sem pivotamento
# usar somente matrizes diagonalmente dominantes

defmodule LU.Kernel1 do
  import GPotion

  gpotion kernel1(a, n, k) do
    i = threadIdx.x + blockIdx.x * blockDim.x

    if (i > k && i < n) do
      a[i * n + k] = a[i * n + k] / a[k * n + k]
    end
  end
end
    
defmodule LU.Kernel2 do
  import GPotion

  gpotion kernel2(a, n, k) do
    i = blockIdx.y * blockDim.y + threadIdx.y
    j = blockIdx.x * blockDim.x + threadIdx.x

    if (i > k && j > k && i < n && j < n) do
      a[i * n + j] = a[i * n + j] - a[i * n + k] * a[k * n + j];
    end
  end
end

# gera matrix diagonalmente dominante a
defmodule MatrixGenerator do
  def generate(size) do
    for i <- 0..(size - 1), into: [] do
      for j <- 0..(size - 1), into: [] do
        if i == j, do: size * 10, else: Enum.random(1..9)
      end
    end
    |> List.flatten()
  end
end

# matrix print para debug
defmodule MatrixPrinter do
  def print_matrex(matrix, size) do
    matrix_list = Matrex.to_list(matrix)

    chunks = Enum.chunk_every(matrix_list, size)

    Enum.each(chunks, fn chunk ->
      IO.puts("[ #{Enum.join(chunk, ", ")} ]")
    end)

    IO.puts("")
  end
end

[arg] = System.argv()
size = String.to_integer(arg)

block_size = 16
#size = 300

#a_matrix = [1, 2, 3, 4, 6, 9, 3, 7, 1]
#a_matrix = [1, 2, 3, 4, 5, 6, 7, 8, 3, 2, 3, 2, 3, 1, 7, 8]
a_matrix = MatrixGenerator.generate(size)

h_a = Matrex.new([a_matrix])

#IO.puts("A matrix:")
#MatrixPrinter.print_matrex(h_a, size)

prev = System.monotonic_time()

kernel1 = GPotion.load(&LU.Kernel1.kernel1/4)
kernel2 = GPotion.load(&LU.Kernel2.kernel2/4)

d_a = GPotion.new_gmatrex(h_a)

dim_block_1 = {block_size, 1, 1}
dim_grid_1 = {div(size + block_size - 1, block_size), 1, 1}

dim_block_2 = {block_size, block_size, 1}
dim_grid_2 = {div(size + block_size - 1, block_size), div(size + block_size - 1, block_size), 1}

# chamada dos kernels
Enum.each(0..(size - 1), fn k -> 
  GPotion.spawn(kernel1, dim_grid_1, dim_block_1, [d_a, size, k])
  GPotion.synchronize()
  GPotion.spawn(kernel2, dim_grid_2, dim_block_2, [d_a, size, k])
  GPotion.synchronize()
end)

_a = GPotion.get_gmatrex(d_a)

next = System.monotonic_time()

#IO.puts("Solution Matrix:")
#MatrixPrinter.print_matrex(a, size)

IO.puts "GPotion\t#{size}\t#{System.convert_time_unit(next-prev,:native,:millisecond)}"