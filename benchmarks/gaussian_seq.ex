defmodule Gaussian.Seq do
  def forward_substitution(matrix, vector) do
    {new_matrix, new_vector} = Enum.reduce(0..length(matrix) - 2, {matrix, vector}, fn i, {m, v} ->
      Enum.reduce((i + 1)..length(m) - 1, {m, v}, fn j, {m_acc, v_acc} ->
        pivot = Enum.at(Enum.at(m_acc, i), i)

        if pivot != 0 do
          factor = Enum.at(Enum.at(m_acc, j), i) / pivot
          new_row = update_row(Enum.at(m_acc, i), Enum.at(m_acc, j), factor)
          new_vector = update_vector(v_acc, i, j, factor)

          # atualiza matriz a com a nova linha
          new_matrix = List.replace_at(m_acc, j, new_row)

          {new_matrix, new_vector}
        else
          {m_acc, v_acc} # não atualiza se pivot é 0
        end
      end)
    end)

    {new_matrix, new_vector}
  end

  def back_substitution(matrix, vector) do
    num_rows = length(matrix)

    # percorre a matrix ao contratrio para fazer a substituição
    Enum.reduce(Enum.to_list(0..num_rows - 1) |> Enum.reverse(), [], fn row_index, acc ->
      row = Enum.at(matrix, row_index)
      var_value = calculate_variable(row, vector, acc, row_index, num_rows)
      [var_value | acc]
    end)
  end

  defp update_row(pivot_row, current_row, factor) do
    # pega duas listas e aplica a função para cada elemento
    Enum.zip_with(pivot_row, current_row, fn pivot, current -> current - pivot * factor end)
  end

  defp update_vector(vector, pivot_index, current_index, factor) do
    pivot_value = Enum.at(vector, pivot_index)
    current_value = Enum.at(vector, current_index)
    # substitui no novo vetor b, a expressão
    List.replace_at(vector, current_index, current_value - pivot_value * factor)
  end

  defp calculate_variable(row, vector, acc, row_index, num_rows) do
    # calcula valor da variável
    sum = Enum.with_index(acc)
            |> Enum.reduce(0, fn {xj, j}, acc_sum -> acc_sum + xj * Enum.at(row, num_rows - j - 1) end)
    (Enum.at(vector, row_index) - sum) / Enum.at(row, row_index)
  end
end

# gera matrix diagonalmente dominante a e vetor resultante b
defmodule MatrixGenerator do
  def generate(size) do
    a = for i <- 0..(size - 1), into: [] do
      for j <- 0..(size - 1), into: [] do
        if i == j, do: size * 10, else: Enum.random(1..9)
      end
    end

    b = Enum.to_list(1..size)

    {a, b}
  end
end

[arg] = System.argv()
size = String.to_integer(arg)

#size = 500
{a, b} = MatrixGenerator.generate(size)

prev = System.monotonic_time()
{upper_matrix, updated_vector} = Gaussian.Seq.forward_substitution(a, b)
next = System.monotonic_time()

_final = Gaussian.Seq.back_substitution(upper_matrix, updated_vector)

#IO.inspect final
IO.puts "Elixir\t#{System.convert_time_unit(next-prev,:native,:millisecond)}"
