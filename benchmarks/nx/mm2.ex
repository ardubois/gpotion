Nx.global_default_backend(EXLA.Backend)


defmodule NxBenchmark.MM do
  import Nx.Defn

  defn mm(t1, t2) do
    Nx.dot(t1, t2)
  end
end

[matrix_size] = System.argv()
matrix_size = String.to_integer(matrix_size)

mat1 = Nx.iota({matrix_size,matrix_size} , type: :f32)

mat2 = Nx.iota({matrix_size,matrix_size}, type: :f32)

IO.inspect mat1

started = System.monotonic_time()

NxBenchmark.MM.mm(mat1, mat2)

finished = System.monotonic_time()

IO.puts "Nx\t#{matrix_size}\t#{System.convert_time_unit(finished-started,:native,:millisecond)} "
