Nx.global_default_backend(EXLA.Backend)

defmodule DataSet do
  def gen_data_set(0), do: []

  def gen_data_set(n) do
    lat = 7 + Enum.random(0..63) + :rand.uniform()
    lon = Enum.random(0..358) + :rand.uniform()
    [[lat, lon] | gen_data_set(n - 1)]
  end
end

defmodule NxBenchmark.NN do
  import Nx.Defn

  defn euclid(tensor, lat, lng) do
    case Nx.shape(tensor) do
      {2} -> :ok
      _ -> raise "invalid shape"
    end

    m_lat = tensor[0]
    m_lng = tensor[1]

    value = Nx.sqrt((lat - m_lat) * (lat - m_lat) + (lng - m_lng) * (lng - m_lng))

    value
  end
end

[size] = System.argv()
size = String.to_integer(size)

t = Nx.tensor(DataSet.gen_data_set(size), type: :f32)
v = Nx.vectorize(t, :coords)

started = System.monotonic_time()
_result = NxBenchmark.NN.euclid(v, 0, 0)
finished = System.monotonic_time()

IO.puts("Elixir\t#{size}\t#{System.convert_time_unit(finished - started, :native, :millisecond)}")
