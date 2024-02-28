# fatoração LU sem pivotamento
# usar somente matrizes diagonalmente dominantes

defmodule LU.Seq do
  def lu_decomposition(matrix) do
    Enum.reduce(0..(length(matrix) - 2), matrix, fn k, acc ->
      acc
      |> update_l(k)
      |> update_u(k)
    end)
  end

  defp update_l(matrix, k) do
    Enum.with_index(matrix, fn row, i ->
      if i > k do
        Enum.with_index(row, fn element, j ->
          if j == k, do: element / Enum.at(Enum.at(matrix, k), k), else: element
        end)
      else
        row
      end
    end)
  end

  defp update_u(matrix, k) do
    Enum.with_index(matrix, fn row, i ->
      if i > k do
        Enum.with_index(row, fn element, j ->
          if j > k, do: element - Enum.at(Enum.at(matrix, i), k) * Enum.at(Enum.at(matrix, k), j), else: element
        end)
      else
        row
      end
    end)
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
  end
end

# matrix print para debug
defmodule MatrixPrinter do
  def print_matrix(matrix) do
    Enum.each(matrix, fn row ->
      IO.puts("[ #{Enum.join(row, ", ")} ]")
    end)
    IO.puts("")
  end
end

[arg] = System.argv()
size = String.to_integer(arg)

#a_matrix = [[1, 2, 3, 4], [5, 6, 7, 8], [3, 2, 3, 2], [3, 1, 7, 8]]
a_matrix = MatrixGenerator.generate(size)

#IO.puts("A matrix:")
#MatrixPrinter.print_matrix(a_matrix)

prev = System.monotonic_time()
_lu_matrix = LU.Seq.lu_decomposition(a_matrix)
next = System.monotonic_time()

#IO.puts("LU Matrix:")
#MatrixPrinter.print_matrix(lu_matrix)
#MatrixPrinter.print_matrex(u, n)

IO.puts "Elixir\t#{size}\t#{System.convert_time_unit(next-prev,:native,:millisecond)}"