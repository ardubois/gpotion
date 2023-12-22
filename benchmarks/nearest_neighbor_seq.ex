require Integer
defmodule DataSet do
  def open_data_set(file) do
    {:ok, contents} = File.read(file)
    contents
    |> String.split("\n", trim: true)
    |> Enum.map(fn f ->  load_file(f) end)
    |> Enum.concat()
    |> Enum.concat()
 #   |> Enum.unzip()
  end
  def load_file(file) do
    #IO.puts file
    {:ok, contents} = File.read(file)
    contents
    |> String.split("\n", trim: true)
    |> Enum.map(fn line -> words = String.split(line, " ", trim: true)
                           [ elem(Float.parse(Enum.at(words, 6)),0), elem(Float.parse(Enum.at(words,7)), 0) ] end  )
  end
  def gen_data_set(0), do: []
  def gen_data_set(n) do
    lat = (7 + Enum.random(0..63)) + :rand.uniform();
      lon = (Enum.random(0..358)) + :rand.uniform();
      [lat,lon|gen_data_set(n-1)]
  end
  def gen_lat_long(_l,c) do
    if(Integer.is_even(c)) do
      (Enum.random(0..358)) + :rand.uniform()
    else
      (7 + Enum.random(0..63)) + :rand.uniform()
    end
  end
end

defmodule NN do
  def euclid_seq([],_lat,_lng) do
    []
  end
  def euclid_seq(array,lat,lng) do
     m_lat = Enum.at(array,0)
     m_lng = Enum.at(array,1)
     value = :math.sqrt((lat-m_lat)*(lat-m_lat)+(lng-m_lng)*(lng-m_lng))
     [value|euclid_seq(Enum.drop(array,2),lat,lng)]
  end


end
#d1 = DataSet.open_data_set("files")
#IO.inspect(d1)

#d1 = DataSet.gen_data_set(100000000)

[arg] = System.argv()

usr_size = String.to_integer(arg)
d1 = DataSet.gen_data_set(usr_size)



prev = System.monotonic_time()
_r_sequential = NN.euclid_seq(d1,0,0)
next = System.monotonic_time()
IO.puts "Elixir\t#{usr_size}\t#{System.convert_time_unit(next-prev,:native,:millisecond)}"


#IO.inspect(m1)
