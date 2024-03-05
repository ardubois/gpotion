Mix.install([{:exla, "~> 0.6.4"}, :matrex])

Nx.global_default_backend(EXLA.Backend)

defmodule NxBenchmark.MM do
  import Nx.Defn

  defn mm(t1, t2) do
    Nx.dot(t1, t2)
  end
end

[matrix_size] = System.argv()
matrix_size = String.to_integer(matrix_size)

gen_random = fn _ -> Enum.random(1..100) end

mat1 =
  Matrex.fill(matrix_size, 1)
  |> Matrex.apply(gen_random)
  |> Matrex.to_list_of_lists()
  |> Nx.tensor(type: :f32)

mat2 =
  Matrex.fill(matrix_size, 1)
  |> Matrex.apply(gen_random)
  |> Matrex.to_list_of_lists()
  |> Nx.tensor(type: :f32)


started = System.monotonic_time()

NxBenchmark.MM.mm(mat1, mat2)

finished = System.monotonic_time()

IO.puts "Nx\t#{matrix_size}\t#{System.convert_time_unit(finished-started,:native,:millisecond)} "
