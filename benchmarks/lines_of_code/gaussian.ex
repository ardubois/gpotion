defmodule Gaussian.Fan1 do
  import GPotion
  gpotion fan1(m, a, size, t) do
    idx = threadIdx.x + blockIdx.x * blockDim.x
    if (idx >= size - 1 - t), do: return
    m[size * (idx + t + 1) + t] = a[size * (idx + t + 1) + t] / a[size * t + t]
  end
end
defmodule Gaussian.Fan2 do
  import GPotion
  gpotion fan2(m, a, b, size, t) do
    xidx = blockIdx.x * blockDim.x + threadIdx.x
    yidx = blockIdx.y * blockDim.y + threadIdx.y
    if (xidx >= size - 1 - t || yidx >= size - t), do: return
    a[size * (xidx + t + 1) + yidx + t] = a[size * (xidx + t + 1) + yidx + t] - m[size * (xidx + t + 1) + t] * a[size * t + yidx + t]
    if (yidx == 0) do
      b[xidx + t + 1] = b[xidx + t + 1] - m[size * (xidx + t + 1) + t] * b[t]
    end
  end
end
defmodule MatrixGenerator do
  def generate(size) do
    a = for i <- 0..(size - 1), into: [] do
      for j <- 0..(size - 1), into: [] do
        if i == j, do: size * 10, else: Enum.random(1..9)
      end
    end
    |> List.flatten()
    b = Enum.to_list(1..size)
    {a, b}
  end
end
defmodule Gaussian.Substitutions do
  def forward_sub(h_a, h_b, h_m, size, max_block_size, block_size_xy) do
    kernel1 = GPotion.load(&Gaussian.Fan1.fan1/4)
    kernel2 = GPotion.load(&Gaussian.Fan2.fan2/5)
    d_a = GPotion.new_gmatrex(h_a)
    d_b = GPotion.new_gmatrex(h_b)
    d_m = GPotion.new_gmatrex(h_m)
    dim_block = {max_block_size, 1, 1}
    dim_grid = {div(size + max_block_size - 1, max_block_size), 1, 1}
    dim_block_xy = {block_size_xy, block_size_xy, 1}
    dim_grid_xy = {div(size + block_size_xy - 1, block_size_xy), div(size + block_size_xy - 1, block_size_xy), 1}
    Enum.each(0..(size - 2), fn t->
      GPotion.spawn(kernel1, dim_grid, dim_block, [d_m, d_a, size, t])
      GPotion.synchronize()
      GPotion.spawn(kernel2, dim_grid_xy, dim_block_xy, [d_m, d_a, d_b, size, t])
      GPotion.synchronize()
    end)
    {GPotion.get_gmatrex(d_a), GPotion.get_gmatrex(d_b)}
  end
  def back_sub(a, b, size) do
    Enum.reduce(0..(size - 1), Matrex.ones(1, size), fn i, acc_vector ->
      sum = if i > 0 do
        Enum.reduce(0..(i - 1), 0, fn j, acc_sum ->
          find_a_diag = Matrex.at(a, 1, size * (size - i - 1) + (size - j))
          find_final_vector_j = Matrex.at(acc_vector, 1, size - j)
          acc_sum + find_a_diag * find_final_vector_j
        end)
      else
        0
      end
      find_b = Matrex.at(b, 1, size - i)
      find_a_diag = Matrex.at(a, 1, size * (size - i - 1) + (size - i))
      Matrex.set(acc_vector, 1, size - i, (find_b - sum) / find_a_diag)
    end)
  end
end
[arg] = System.argv()
size = String.to_integer(arg)
max_block_size = 512
block_size_xy = 4
{a_matrix, b_vector} = MatrixGenerator.generate(size)
h_a = Matrex.new([a_matrix])
h_b = Matrex.new([b_vector])
h_m = Matrex.zeros(1, size * size)
prev = System.monotonic_time()
{a, b} = Gaussian.Substitutions.forward_sub(h_a, h_b, h_m, size, max_block_size, block_size_xy)
next = System.monotonic_time()
_final_vector = Gaussian.Substitutions.back_sub(a, b, size)
IO.puts "GPotion\t#{size}\t#{System.convert_time_unit(next-prev,:native,:millisecond)}"
